import unittest
import psh
import numpy as np

class TestIndexFunctions(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(0)
        self.d=2
        self.width=10000
        self.shape=(self.width,)*self.d
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
            self.assertEqual(psh.point_to_index(pt,self.shape),self._point_to_index(pt,self.width,self.max))
    def test_index_to_point_unsigned(self):
        pts=self.rng.integers(0,self.max,size=(1000,),dtype=np.uint64)
        for pt in pts:
            mine=tuple(psh.index_to_point(pt,self.shape))
            theirs=tuple(self._index_to_point(pt,self.width,self.max,self.d))
            self.assertEqual(mine,theirs)

    def test_point_to_index_to_point_unsigned(self):
        pts=self.rng.integers(0,self.width,size=(1000,self.d),dtype=np.uint64)
        for pt in pts:
            index=psh.point_to_index(pt,self.shape)
            self.assertEqual(tuple(psh.index_to_point(index,self.shape)),tuple(pt))

    def test_index_to_point_negative(self):
        pts=self.rng.integers(0,self.max,size=(1000,),dtype=np.uint64)
        for pt in pts:
            mine=tuple(psh.index_to_point(pt,self.shape))
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
                    correct=self._hash_function(pt,prime,parameter)
                    mine=psh.entry_hash(pt,prime,parameter)
                    self.assertEqual(mine,correct)

class TestRandomData(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(0)
        self.width=48
        self.d=2
        self.shape=(self.width,)*self.d
        self.data=[]
        self.data_b=np.full(self.width*self.width,False)
        count=0
        for x in range(self.width):
            for y in range(self.width):
                if self.rng.uniform()<(1/2):
                    element=psh.data_tuple(np.array([x,y],dtype=np.uint64),count)
                    count+=1
                    self.data.append(element)
                    self.data_b[psh.point_to_index(element.location,self.shape)]=True
        self.hashmap=psh.PerfectSpatialHashMap(self.data,self.d,self.width,0)
        self.maxint=np.iinfo(self.hashmap.int_type).max
    def test_all_pts_exist(self):
        for i in np.arange(self.width**self.d,dtype=self.hashmap.int_type):
            if self.data_b[i]:
                p=psh.index_to_point(i,self.shape).astype(self.hashmap.int_type)
                self.hashmap[p]
    def test_no_imaginary_pts(self):
        for i in np.arange(self.width**self.d,dtype=self.hashmap.int_type):
            if not self.data_b[i]:
                p=psh.index_to_point(i,self.shape).astype(self.hashmap.int_type)
                self.assertRaises(KeyError,
                    self.hashmap.__getitem__,
                    p)
    def test_all_pts_correct(self):
        for i in np.arange(self.width**self.d,dtype=self.hashmap.int_type):
            if self.data_b[i]:
                p=psh.index_to_point(i,self.shape).astype(self.hashmap.int_type)
                self.assertEqual(self.data[i].contents,self.hashmap[p])
class Test3DRandomData(TestRandomData):
    def setUp(self):
        self.rng=np.random.default_rng(0)
        self.width=12
        self.d=3
        self.shape=(self.width,)*self.d
        self.data=[]
        self.data_b=np.full(self.width**self.d,False)
        count=0
        for x in range(self.width):
            for y in range(self.width):
                for z in range(self.width):
                    if self.rng.uniform()<(1/2):
                        element=psh.data_tuple(np.array([x,y,z],dtype=np.uint64),count)
                        count+=1
                        self.data.append(element)
                        self.data_b[psh.point_to_index(element.location,self.shape)]=True
        self.hashmap=psh.PerfectSpatialHashMap(self.data,self.d,self.width,0)
        self.maxint=np.iinfo(self.hashmap.int_type).max


if __name__=="__main__":
    unittest.main()