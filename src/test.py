import unittest
import psh
import numpy as np

class TestIndexFunctions(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(0)
        self.d=2
        self.width=10000
        self.max=int(self.width**self.d)
        self.maxint=np.iinfo(np.array([self.width]).dtype).max
    def _point_to_index(self,point,width,max):
        d=len(point)
        index=point[-1]
        for pt in reversed(point[:-1]):
            index+=width*pt
            width*=width
        return int(index % max)
    def _index_to_point(self,index,width,max,d):
        if d==1:
            return np.array([index%width])
        elif d==2:
            return np.array([index//width,index%width])
        elif d==3:
            return np.array([index//(width*width),
            index%(width*width)//width,
            (index % (width*width))%width])
        else:
            output=np.full(d,index)
            max/=width
            for i in range(d):
                output[i]=index/max
                index%=max
                if (i+1)<d:
                    max/=width
            return output

    def test_point_to_index_unsigned(self):
        pts=self.rng.integers(0,self.width,size=(1000,self.d),dtype=np.uint64)
        for pt in pts:
            with self.subTest(pt=pt):
                self.assertEqual(psh.point_to_index(pt,self.width,self.d),self._point_to_index(pt,self.width,self.max))
    def test_index_to_point_unsigned(self):
        pts=self.rng.integers(0,self.max,size=(1000,),dtype=np.uint64)
        for pt in pts:
            with self.subTest(index=pt):
                mine=tuple(psh.index_to_point(pt,self.width,self.d))
                theirs=tuple(self._index_to_point(pt,self.width,self.max,self.d))
                self.assertEqual(mine,theirs)

    def test_point_to_index_to_point_unsigned(self):
        pts=self.rng.integers(0,self.width,size=(1000,self.d),dtype=np.uint64)
        for pt in pts:
            with self.subTest(pt=pt):
                index=psh.point_to_index(pt,self.width,self.d)
                self.assertEqual(tuple(psh.index_to_point(index,self.width,self.d)),tuple(pt))

    def test_index_to_point_negative(self):
        pts=self.rng.integers(0,self.max,size=(1000,),dtype=np.uint64)
        for pt in pts:
            with self.subTest(index=pt):
                mine=tuple(psh.index_to_point(pt,self.width,self.d,self.maxint))
                theirs=tuple(self._index_to_point(pt,self.width,self.maxint,self.d))
                self.assertEqual(mine,theirs)
class TestEntryHash(unittest.TestCase):
    def setUp(self) -> None:
        self.primes=[3, 97, 193, 389, 769, 1543, 3079,
				6151, 12289, 24593, 49157, 98317, 196613, 393241, 786433, 1572869,
				3145739, 6291469]
        self.parameters=list(range(10))
        self.rng=np.random.default_rng(0)
        self.d=2
        self.width=10000
        self.max=int(self.width**self.d)
        self.pts=self.rng.integers(0,self.width,size=(1000,self.d),dtype=np.uint64)
    def _hash_function(self,point,prime,parameter):
        return prime*np.dot(point,parameter**(np.arange(self.d)+1))
    def test_entry_hash(self):
        for prime in self.primes:
            for pt in self.pts:
                for parameter in self.parameters:
                    with self.subTest(prime=prime,point=pt,k=parameter):
                        correct=self._hash_function(pt,prime,parameter)
                        mine=psh.entry_hash(pt,prime,parameter)
                        self.assertEqual(mine,correct)
if __name__=="__main__":
    unittest.main()