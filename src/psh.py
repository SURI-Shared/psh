from math import ceil
from collections import defaultdict,namedtuple
import numpy as np
from sys import getsizeof

from numba import njit,typeof,types
from numba.experimental import jitclass

data_tuple=namedtuple("DataTuple",["location","contents"])
numba_data_tuple_type=typeof(data_tuple)

@njit
def point_to_index(point,shape):
    return (point[-1]+np.sum(point[:-1]*np.cumprod(shape[:0:-1])[::-1]))%np.prod(shape)

@njit
def index_to_point(index,shape):
    stride=np.concatenate((np.cumprod(shape[-1:0:-1])[::-1],np.array([1],dtype=shape.dtype)))
    output=np.empty_like(shape)
    for i in range(len(shape)):
        output[i]=index//stride[i]
        index%=stride[i]
    return output

@njit
def entry_hash(point,prime,hash_parameter):
    '''
    compute hash of a point with a particular positional parameter

    Parameters: point : (d,) np integer array
                    location in the domain
                prime : integer
                    prime number to use in the hashing
                hash_parameter: integer
                    (small) integer to use in hashing to avoid collisions
    Returns:    hk : integer
                    integer hash of point 
    '''
    return np.sum(point*(hash_parameter**np.arange(1,len(point)+1,dtype=point.dtype)))*prime

def create_psh(data,domain_dim,domain_limit,seed,verbose=False):
    int_type=type(data[0].location[0])
    numba_int_type=typeof(data[0].location[0])
    #TODO: convert to subclass of collections.abc.MutableMapping
    class PerfectSpatialHashMap:
        def __init__(self,data,domain_dim,domain_limit,seed,verbose=False) -> None:
            self.verbose=verbose
            #input integer type
            #number of data points
            self.n=len(data)
            #dimension of data
            self.d=int_type(domain_dim)
            #random number generator
            self.generator=np.random.default_rng(seed)
            #pick three primes for use in hashing
            self.primes=[3, 97, 193, 389, 769, 1543, 3079,
                    6151, 12289, 24593, 49157, 98317, 196613, 393241, 786433, 1572869,
                    3145739, 6291469]
            self.M0=self.prime()
            self.M1=self.prime()
            while self.M1==self.M0:
                self.M1=self.prime()
            self.M2=self.prime()
            #width of the hash table
            self.m_bar=int_type(ceil(self.n**(1/self.d)))
            #size of the hash table
            self.m=self.m_bar**self.d
            #width of the offset table (will be updated)
            self.r_bar=int_type(ceil((self.n/self.d)**(1/self.d))-1)
            #u_bar is the limit of the domain in each dimension
            try:
                #can specify as an iterable for the size of each dimension
                l=len(domain_limit)
            except TypeError:
                #or a singleton in which case all dims have same size
                l=1
            if l==self.d:
                #was passed domain extents in each dimension
                self.u_bar=domain_limit
            elif l==1:
                self.u_bar=np.array((domain_limit,)*self.d)
            else:
                raise ValueError("Length of non-singleton domain_limit doesn't match domain_dim") 
            #u is the number of elements in the domain
            self.u=np.prod(self.u_bar)
            #list of entries in the hash table, see PerfectSpatialHashEntry
            self.H=[]
            #create the offset table
            create_succeeded=False
            while not create_succeeded:
                self.r_bar+=self.d
                #size of the offset table
                self.r=self.r_bar**self.d
                #initial offset table array
                if self.verbose:
                    print("Try offset table of width "+str(self.r_bar))
                self.phi=np.zeros((self.r,self.d),dtype=int_type)
                create_succeeded=self.try_create_hash_table(data)

        def memory_size(self):
            internal_size=sum((getsizeof(item) for item in self.__dict__.values()))
            hash_table_size=sum((getsizeof(Hi) for Hi in self.H))
            return internal_size+hash_table_size
        def prime(self):
            return self.generator.choice(self.primes)

        def try_create_hash_table(self,data):
            H_hat=[PerfectSpatialHashEntry()]*self.m
            H_b_hat=np.full(self.m,False)
            if self.bad_m_r():
                return False
            
            #find out what order we should do the hashing to optimize success rate
            buckets = self.create_buckets(data);
            if self.verbose:
                print("Jiggling offsets")
            for i,bucket in enumerate(buckets):
                #if a bucket is empty, then the rest will also be empty
                if len(bucket)== 0:
                    break

                #try to jiggle the offsets until an injective mapping is found
                if (not self.jiggle_offsets(H_hat, H_b_hat, bucket)):
                    return False
            if self.verbose:
                print("Tweaking Positional Hash")
            if not self.hash_positions(data, H_hat):
                return False
            if self.verbose:
                print("Success")
            self.H=H_hat
            return True

        def bad_m_r(self):
            '''
            test if hash table and offset table widths are not coprime or are an empirically bad case
            '''
            m_mod_r=self.m_bar%self.r_bar
            return m_mod_r==1 or m_mod_r==(self.r_bar-1) or np.gcd(self.m_bar,self.r_bar)>1
            
        def create_buckets(self,data):
            buckets=[Bucket(i) for i in range(self.r)]
            for element in data:
                h1=element.location*self.M1
                buckets[self.point_to_offset_idx(h1)].append(element)
            buckets.sort(reverse=True)#sort buckets in descending order of size
            return buckets

        def jiggle_offsets(self,H_hat,H_b_hat,bucket):
            #start at a random point
            start_offset=self.generator.integers(0,self.m).astype(int_type)

            found=False
            found_offset=np.ones((self.d,),dtype=int_type)
            #try every possible offset
            for i in np.arange(self.m,dtype=int_type):
                #wrap around by the size of the hash table, then convert to a multi index
                phi_offset=self.index_to_hash_pt(((start_offset+i)%self.m).astype(int_type))
                collision=False
                #now check every element in the bucket for collisions
                for element in bucket:
                    h0=element.location*self.M0
                    h1=element.location*self.M1
                    index=self.point_to_offset_idx(h1)
                    #apply existing offset (phi_hat) everywhere except the current index
                    if index==bucket.phi_index:
                        offset=phi_offset
                    else:
                        offset=self.phi[index]
                    hash=h0+offset
                    #check if the resulting hash is in use
                    collision=H_b_hat[self.point_to_hash_idx(hash)]
                    if collision:
                        break
                if not collision:
                    found=True
                    found_offset=phi_offset
                    break
            if found:
                #update the offset table array with the new offset
                self.phi[bucket.phi_index]=found_offset
                #and insert the bucket into the hash table at that location
                self.insert(bucket,H_hat,H_b_hat)
                return True
            else:
                return False

        def insert(self,bucket,H_hat,H_b_hat):
            for element in bucket:
                i=self.get_item_index(element.location)
                H_hat[i]=PerfectSpatialHashEntry(element,self.M2)
                H_b_hat[i]=True

        def hash(self,point,offset=None):
            '''
            return the index in the hash table for a given position in the domain

            Parameters: point : (d,) np integer array
                            location in the domain
                        offset: integer or None (default)
                            if None, looks up offset in self.phi
            Returns:    hash_position: integer
                            index into hash table (which is stored at self.H)
            '''
            h0=point*self.M0
            h1=point*self.M1
            i=self.point_to_offset_idx(h1)
            if offset is None:
                offset=self.phi[i]
            return h0+offset

        def hash_positions(self,data,H_hat):
            indices=np.full(self.m,False)
            data_b=np.full(self.u,False)
            for element in data:
                data_b[self.point_to_domain_idx(element.location)]=True
            #sweep thru all points in domain without a data entry
            for i in range(self.u):
                if data_b[i]:
                    continue
                p=self.index_to_domain_pt(i)
                l=self.get_item_index(p)
                #record if position hash collides with existing element
                if H_hat[l].equals(p,self.M2):
                    indices[l]=True
            #go through stored collisions and record all points that map to those indices
            collisions=defaultdict(list)
            for i in range(self.u):
                p=self.index_to_domain_pt(i)
                l=self.get_item_index(p)
                if indices[l]:
                    collisions[l].append(i)
            #try to change positional hash parameter until no collisions
            success=True
            for domain_index,colliding_indices in collisions.items():
                if not self.fix_k(H_hat[domain_index],colliding_indices):
                    success=False
            return success
        def fix_k(self,H_entry,colliding_indices):
            '''
            recursively tweak hash parameter at a position until collisions are resolved

            Parameters: H_entry : PerfectSpatialHashEntry 
                            the PerfectSpatialHashEntry experiencing collisions
                        colliding_indices : list of integers
                            the 1D (domain index) positions that all map to H_entry
            Returns:    success
                            calls itself recursively until a H_entry.k is found that doesn't collide OR all have been tried  
            '''
            maxint=np.iinfo(int_type).max
            H_entry.increment_k(self.M2)
            #if k is 0, all values have been tried
            if H_entry.k==0:
                return False
            success=True
            for i in colliding_indices:
                #i is a domain index
                p=self.index_to_domain_pt(i)
                hk=entry_hash(p, self.M2, H_entry.k)
                if np.any(H_entry.location!=p) and H_entry.hk==hk:
                    success=False
                    break
            #if the k didin't work, recurse
            if not success:
                return self.fix_k(H_entry,colliding_indices)
            return True
        def point_to_domain_idx(self,point):
            return self._point_to_index(point,self.u_bar)
        def point_to_hash_idx(self,point):
            return self._point_to_index(point,np.array((self.m_bar,)*self.d))
        def point_to_offset_idx(self,point):
            return self._point_to_index(point,np.array((self.r_bar,)*self.d))
        def _point_to_index(self,point,shape):
            return int_type(point_to_index(point,shape))
        def index_to_domain_pt(self,index):
            return self._index_to_point(index,self.u_bar)
        def index_to_hash_pt(self,index):
            return self._index_to_point(index,np.array((self.m_bar,)*self.d))
        def index_to_offset_pt(self,index):
            return self._index_to_point(index,np.array((self.r_bar,)*self.d))
        def _index_to_point(self,index,shape):
            return index_to_point(index,shape).astype(int_type)

        def get_offset_table_index(self,point):
            '''
            compute the index into the offset table. Implements h1 with wrapping
            '''
            return self.point_to_offset_idx(point*self.M1)
        def get_hash_table_index(self,point):
            '''
            compute the index into the hash table (with NO offset). Implements h0 with wrapping.
            '''
            return self.point_to_hash_idx(point*self.M0)
        def get_item_index(self,point):
            '''
            compute the index into the hash table WITH offset. Use for data access
            '''
            return self.point_to_hash_idx(self.hash(point))
        def __getitem__(self,point):
            #find where element would be located
            i=self.get_item_index(point)
            #and check has correct positional hash
            if self.H[i].equals(point,self.M2):
                return self.H[i].contents
            else:
                raise KeyError("Element not found in map")
        def __setitem__(self,point,contents):
            i=self.get_item_index(point)
            if self.H[i].hk is None:
                #nothing exists at this location
                self.H[i]=PerfectSpatialHashEntry(data_tuple(point,contents),self.M2)
            elif self.H[i].equals(point,self.M2):
                #overwrite contents
                self.H[i].update(point,contents,self.M2)
            else:
                #failure
                raise ValueError("Unable to add element to map")
        def count_real_entries(self):
            return sum((len(e.location)==self.d for e in self.H))
    class Bucket(list):
        def __init__(self,offset_index):
            self.phi_index=offset_index
            super().__init__()
        def __lt__(self,rhs):
            return len(self)<len(rhs)

    @jitclass({
        "k":numba_int_type,
        "hk":numba_int_type,
        "contents":typeof(data[0].contents),
        "location":typeof(data[0].location)
        })
    class PerfectSpatialHashEntry:
        def __init__(self,data_element=None,prime=None):
            self.k=1
            if data_element is not None:
                self.update(data_element.location,data_element.contents,prime)
            else:
                self.hk=1
        def rehash(self,point,prime,new_k=1):
            self.k=new_k
            self.hk=entry_hash(point,prime,self.k)
        def increment_k(self,prime):
            self.rehash(self.location,prime,self.k+1)
        def equals(self,point,prime):
            return self.hk==entry_hash(point,prime,self.k)
        def update(self,location,contents,prime):
            self.contents=contents
            self.location=location
            self.rehash(self.location,prime,self.k)
        def __sizeof__(self):
            return sum((getsizeof(item) for item in self.__dict__.values()))
    return PerfectSpatialHashMap(data,domain_dim,domain_limit,seed,verbose)