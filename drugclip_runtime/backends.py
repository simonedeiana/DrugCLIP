"""Run a portable DrugCLIP encoder on already-preprocessed tensor inputs.
Usage: python infer.py molecule.onnx molecule-example.npz embedding.npy
Use a .engine model for NVIDIA TensorRT instead of ONNX Runtime CPU.
"""
from pathlib import Path
import argparse
import numpy as np
from collections import OrderedDict

NAMES = ('tokens','distances','edge_types')

class OnnxEncoder:
    def __init__(self, path, threads=6):
        import onnxruntime as ort
        opt=ort.SessionOptions()
        opt.intra_op_num_threads=threads
        self.session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])
    def __call__(self, inputs):
        return self.session.run(['embedding'],{k:np.ascontiguousarray(inputs[k]) for k in NAMES})[0]

class TensorRTEncoder:
    _logger=None
    def __init__(self, path):
        import tensorrt as trt
        from cuda.bindings import runtime as cuda
        self.cuda,self.trt=cuda,trt
        if TensorRTEncoder._logger is None:
            TensorRTEncoder._logger=trt.Logger(trt.Logger.WARNING)
        self.logger=TensorRTEncoder._logger
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None: raise RuntimeError('Cannot deserialize TensorRT engine on this platform/device')
        self.context=self.engine.create_execution_context()
        if self.context is None: raise RuntimeError('Cannot create TensorRT execution context')
        self.stream=self.check(cuda.cudaStreamCreate())
        self.buffers={}
    @staticmethod
    def check(result):
        if int(result[0]) != 0: raise RuntimeError(f'CUDA error: {result[0]}')
        return result[1] if len(result)==2 else result[1:]
    def prepare(self, inputs):
        c=self.cuda
        self.host={}
        for k in NAMES:
            a=np.ascontiguousarray(inputs[k],dtype=self.trt.nptype(self.engine.get_tensor_dtype(k)))
            if not self.context.set_input_shape(k,a.shape): raise ValueError(f'Unsupported shape for {k}: {a.shape}')
            self.host[k]=a
        self.host['embedding']=np.empty(tuple(self.context.get_tensor_shape('embedding')),dtype=self.trt.nptype(self.engine.get_tensor_dtype('embedding')))
        for k,a in self.host.items():
            if k not in self.buffers or self.buffers[k][1]<a.nbytes:
                if k in self.buffers: self.check(c.cudaFree(self.buffers[k][0]))
                self.buffers[k]=(self.check(c.cudaMalloc(a.nbytes)),a.nbytes)
            ptr=self.buffers[k][0]
            if not self.context.set_tensor_address(k,ptr): raise RuntimeError(k)
            if k!='embedding': self.check(c.cudaMemcpy(ptr,a.ctypes.data,a.nbytes,c.cudaMemcpyKind.cudaMemcpyHostToDevice))
    def enqueue(self):
        if not self.context.execute_async_v3(self.stream): raise RuntimeError('TensorRT execution failed')
    def synchronize(self): self.check(self.cuda.cudaStreamSynchronize(self.stream))
    def __call__(self,inputs):
        self.prepare(inputs)
        self.enqueue()
        self.synchronize()
        a=self.host['embedding']
        self.check(self.cuda.cudaMemcpy(a.ctypes.data,self.buffers['embedding'][0],a.nbytes,self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return a.copy()
    def close(self):
        if self.stream is None: return
        self.synchronize()
        for ptr,_ in self.buffers.values(): self.check(self.cuda.cudaFree(ptr))
        self.buffers.clear()
        self.check(self.cuda.cudaStreamDestroy(self.stream))
        self.stream=None

class CudaGraphEncoder:
    """Bounded shape-specific graph cache; each graph owns its context and buffers.

    Copies remain outside capture so caller-owned NumPy addresses may change.
    Precision and engine tactics are unchanged. Capture errors are explicit.
    """
    def __init__(self, path, max_graphs=16):
        if max_graphs < 1: raise ValueError('max_graphs must be positive')
        self.owner=TensorRTEncoder(path)
        self.max_graphs=max_graphs
        self.cache=OrderedDict()
        self.captures=0
        self.hits=0
        self.evictions=0

    def _destroy(self, entry):
        encoder,graph,instance=entry
        encoder.synchronize()
        encoder.check(encoder.cuda.cudaGraphExecDestroy(instance))
        encoder.check(encoder.cuda.cudaGraphDestroy(graph))
        encoder.close()

    def __call__(self, inputs):
        key=tuple(tuple(inputs[k].shape) for k in NAMES)
        if key in self.cache:
            encoder,graph,instance=self.cache.pop(key)
            self.hits+=1
            for k in NAMES:
                a=np.ascontiguousarray(inputs[k],dtype=encoder.trt.nptype(encoder.engine.get_tensor_dtype(k)))
                encoder.host[k]=a
                encoder.check(encoder.cuda.cudaMemcpyAsync(encoder.buffers[k][0],a.ctypes.data,a.nbytes,encoder.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,encoder.stream))
        else:
            if len(self.cache)>=self.max_graphs:
                self._destroy(self.cache.popitem(last=False)[1]);self.evictions+=1
            encoder=object.__new__(TensorRTEncoder)
            for attr in ('cuda','trt','logger','runtime','engine'):
                setattr(encoder,attr,getattr(self.owner,attr))
            encoder.context=encoder.engine.create_execution_context()
            if encoder.context is None: raise RuntimeError('Cannot create graph context')
            c=encoder.cuda
            encoder.stream=encoder.check(c.cudaStreamCreateWithFlags(1))
            encoder.buffers={}
            encoder.prepare(inputs)
            encoder.check(c.cudaDeviceSynchronize())
            encoder.enqueue();encoder.synchronize()
            encoder.check(c.cudaStreamBeginCapture(encoder.stream,c.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
            try:
                encoder.enqueue()
            finally:
                result=c.cudaStreamEndCapture(encoder.stream)
            graph=encoder.check(result)
            instance=encoder.check(c.cudaGraphInstantiate(graph,0))
            self.captures+=1
        self.cache[key]=(encoder,graph,instance)
        encoder.check(encoder.cuda.cudaGraphLaunch(instance,encoder.stream))
        encoder.synchronize()
        a=encoder.host['embedding']
        encoder.check(encoder.cuda.cudaMemcpy(a.ctypes.data,encoder.buffers['embedding'][0],a.nbytes,encoder.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return a.copy()

    def close(self):
        for entry in self.cache.values(): self._destroy(entry)
        self.cache.clear()
        self.owner.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model');p.add_argument('inputs');p.add_argument('output')
    a=p.parse_args()
    encoder=TensorRTEncoder(a.model) if a.model.endswith('.engine') else OnnxEncoder(a.model)
    with np.load(a.inputs) as inputs: output=encoder(inputs)
    if not np.isfinite(output).all(): raise RuntimeError('Non-finite output')
    np.save(a.output,output)
    print('Saved normalized embeddings:',output.shape)
    if isinstance(encoder,TensorRTEncoder): encoder.close()
