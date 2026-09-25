import unittest
from src.short_audio import partition,RATE

class PartitionTests(unittest.TestCase):
    def test_contiguous_cores_and_overlap(self):
        owners=[{'id':1,'start':0.,'end':22.},{'id':2,'start':22.,'end':44.}]
        rows=partition(owners,44*RATE,[])
        self.assertEqual(sum(r['core_end_frame']-r['core_start_frame'] for r in rows),44*RATE)
        self.assertEqual(rows[0]['crop_start_frame'],0)
        self.assertEqual(rows[-1]['crop_end_frame'],44*RATE)
        self.assertTrue(all(2*RATE<=r['core_end_frame']-r['core_start_frame']<=8*RATE for r in rows))
        self.assertTrue(all(a['core_end_frame']==b['core_start_frame'] for a,b in zip(rows,rows[1:])))
        self.assertTrue(all(r['crop_start_frame']<=r['core_start_frame']<r['core_end_frame']<=r['crop_end_frame'] for r in rows))
    def test_vad_only_moves_cut_nearest_target(self):
        speech=[{'start':0,'end':5*RATE},{'start':7*RATE,'end':10*RATE}]
        rows=partition([{'id':1,'start':0.,'end':20.}],20*RATE,speech)
        self.assertEqual(rows[0]['core_end_frame'],6*RATE)
        self.assertEqual(sum(r['core_end_frame']-r['core_start_frame'] for r in rows),20*RATE)
    def test_tail_balanced(self):
        rows=partition([{'id':1,'start':0.,'end':9.}],9*RATE,[])
        self.assertEqual([r['core_end_frame']-r['core_start_frame'] for r in rows],[6*RATE,3*RATE])
    def test_reject_gap(self):
        with self.assertRaises(ValueError):partition([{'id':1,'start':0.,'end':20.},{'id':2,'start':21.,'end':40.}],40*RATE,[])
    def test_reject_budget(self):
        with self.assertRaises(ValueError):partition([{'id':1,'start':0.,'end':4000.}],4000*RATE,[])
    def test_reject_invalid_vad(self):
        with self.assertRaises(ValueError):partition([{'id':1,'start':0.,'end':20.}],20*RATE,[{'start':10,'end':9}])

if __name__=='__main__':unittest.main()
