import unittest
import numpy as np
from drugclip_runtime.data import prepare, collate


class PreprocessingTests(unittest.TestCase):
    def test_hydrogen_special_tokens_and_distances(self):
        item = prepare(dict(atoms=['C', 'H', 'N'], coordinates=np.array([[0, 0, 0], [1, 0, 0], [3, 4, 0]])), 'molecule')
        np.testing.assert_array_equal(item['tokens'], [1, 4, 5, 2])
        self.assertEqual(item['distances'][1, 2], 5)
        self.assertEqual(item['distances'][0].sum(), 0)
        batch = collate([item])
        self.assertEqual(batch['tokens'].shape, (1, 8))
        self.assertEqual(batch['edge_types'][0, 1, 2], 4 * 31 + 5)
        self.assertEqual(batch['edge_types'][0, 4:].sum(), 0)

    def test_conformer_selection_and_invalid_input(self):
        record = dict(atoms=['C', 'N'], coordinates=np.array([[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [2, 0, 0]]]))
        self.assertEqual(prepare(record, 'molecule', conformer=1)['distances'][1, 2], 2)
        with self.assertRaises(ValueError):
            prepare(record, 'molecule', conformer=2)
        with self.assertRaises(ValueError):
            prepare(dict(atoms=['H'], coordinates=[[0, 0, 0]]), 'molecule')

    def test_crop_is_deterministic(self):
        record = dict(pocket_atoms=['CA'] * 20, pocket_coordinates=np.arange(60).reshape(20, 3))
        a = prepare(record, 'pocket', max_pocket_atoms=5)
        b = prepare(record, 'pocket', max_pocket_atoms=5)
        np.testing.assert_array_equal(a['distances'], b['distances'])
        self.assertEqual(len(a['tokens']), 7)


if __name__ == '__main__':
    unittest.main()
