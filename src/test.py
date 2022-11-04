import unittest
import psh
import numpy as np

class TestIndexFunctions(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(0)
        self.d=2
        self.width=10000
        self.shape=np.array((self.width,)*self.d)
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
            self.assertEqual(psh.point_to_index(pt,self.shape.astype(np.uint64)),self._point_to_index(pt,self.width,self.max))
    def test_index_to_point_unsigned(self):
        pts=self.rng.integers(0,self.max,size=(1000,),dtype=np.uint64)
        for pt in pts:
            mine=tuple(psh.index_to_point(pt,self.shape.astype(np.uint64)))
            theirs=tuple(self._index_to_point(pt,self.width,self.max,self.d))
            self.assertEqual(mine,theirs)

    def test_point_to_index_to_point_unsigned(self):
        pts=self.rng.integers(0,self.width,size=(1000,self.d),dtype=np.uint64)
        for pt in pts:
            index=psh.point_to_index(pt,self.shape.astype(np.uint64))
            self.assertEqual(tuple(psh.index_to_point(index,self.shape.astype(np.uint64))),tuple(pt))

    def test_index_to_point_negative(self):
        pts=self.rng.integers(0,self.max,size=(1000,),dtype=np.uint64)
        for pt in pts:
            mine=tuple(psh.index_to_point(pt,self.shape.astype(np.uint64)))
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
        self.shape=np.array((self.width,)*self.d,dtype=np.uint64)
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
    def test_all_pts_correct_contents(self):
        msg="Contents at {0}={1}; should be {2}"
        for element in self.data:
            contents=self.hashmap[element.location]
            true_contents=element.contents
            self.assertEqual(true_contents,contents,msg=msg.format(element.location,contents,true_contents))
    def test_all_pts_correct_location(self):
        msg="Location should be {0}, is {1}"
        for element in self.data:
            idx=self.hashmap.get_item_index(element.location)
            if self.hashmap.H[idx].equals(element.location,self.hashmap.M2):
                true_location=element.location
                location=self.hashmap.H[idx].location
                self.assertTrue(np.all(true_location==location),msg.format(true_location,location))
    def test_bucket_offset_indices(self):
        buckets=self.hashmap.create_buckets(self.data)
        for bucket in buckets:
            for element in bucket:
                self.assertEqual(self.hashmap.get_offset_table_index(element.location),bucket.phi_index)
    def test_bucket_item_indices_unique(self):
        buckets=self.hashmap.create_buckets(self.data)
        for bucket in buckets:
            indices=set()
            for element in bucket:
                i=self.hashmap.get_item_index(element.location)
                self.assertNotIn(i,indices)
                indices.add(i)
    def test_entry_count(self):
        self.assertEqual(self.hashmap.count_real_entries(),len(self.data))
    def test_all_entries_real(self):
        '''
        test that every entry in hash table is also in the data set
        '''
        locations=np.array([e.location for e in self.data])
        for entry in self.hashmap.H:
            if entry.location is not None:
                logical_idx=np.all(entry.location==locations,1)
                self.assertTrue(np.any(logical_idx),msg="Hashmap contains location "+str(entry.location)+" and shouldn't")
                idx=np.nonzero(logical_idx)[0][0]
                self.assertEqual(entry.contents,self.data[idx].contents)
    def test_no_entries_missing(self):
        '''
        test that every entry in the data set is in the hash table
        '''
        unused=set(range(len(self.hashmap.H)))
        for element in self.data:
            found=False
            for idx in unused:
                if np.all(self.hashmap.H[idx].location==element.location):
                    if self.hashmap.H[idx].contents==element.contents:
                        found=True
                        unused.remove(idx)
                        break
            self.assertTrue(found,msg="element "+str(element)+" not found in hash table")

class Test3DRandomData(TestRandomData):
    def setUp(self):
        self.rng=np.random.default_rng(0)
        self.width=12
        self.d=3
        self.shape=np.array((self.width,)*self.d,dtype=np.uint64)
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