"""Run a portable DrugCLIP encoder on already-preprocessed tensor inputs.
Usage: python infer.py molecule.onnx molecule-example.npz embedding.npy
Use a .engine model for NVIDIA TensorRT instead of ONNX Runtime CPU.
"""
from pathlib import Path
import argparse
import numpy as np

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
