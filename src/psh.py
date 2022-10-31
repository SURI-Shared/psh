from math import ceil
from collections import defaultdict,namedtuple
import numpy as np

data_tuple=namedtuple("DataTuple",["location","contents"])

#TODO: convert to subclass of collections.abc.MutableMapping
class PerfectSpatialHashMap:
    def __init__(self,data,domain_dim,domain_limit,seed) -> None:
        #number of data points
        self.n=len(data)
        #dimension of data
        self.d=domain_dim
        #input integer type
        self.int_type=data[0].location.dtype
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
        self.m_bar=ceil(self.n**(1/self.d))
        #size of the hash table
        self.m=self.m_bar**self.d
        #width of the offset table (will be updated)
        self.r_bar=ceil((self.n/self.d)**(1/self.d))-1
        #u_bar is the limit of the domain in each dimension
        self.u_bar=domain_limit
        #u is the number of elements in the domain
        self.u=domain_limit**self.d
        #list of entries in the hash table, see PerfectSpatialHashEntry
        self.H=[]
        #create the offset table
        create_succeeded=False
        while not create_succeeded:
            self.r_bar+=self.d
            #size of the offset table
            self.r=self.r_bar**self.d
            #initial offset table array
            self.phi=np.empty((self.r,self.d),dtype=self.int_type)
            create_succeeded=self.try_create_hash_table(data)

    def prime(self):
        return self.generator.choice(self.primes)

    def try_create_hash_table(self,data):
        H_hat=[PerfectSpatialHashEntry()]*self.m
        H_b_hat=np.full(self.m,False)
        if self.bad_m_r():
            return False
        
        #find out what order we should do the hashing to optimize success rate
        buckets = self.create_buckets(data);
        
        for bucket in buckets:
            #if a bucket is empty, then the rest will also be empty
            if len(bucket)== 0:
                break

            #try to jiggle the offsets until an injective mapping is found
            if (not self.jiggle_offsets(H_hat, H_b_hat, self.phi, bucket)):
                return False

        if not self.hash_positions(data, H_hat):
            return False
        self.H=H_hat
        return True

    def bad_m_r(self):
        '''
        test if hash table and offset table widths are coprime
        '''
        m_mod_r=self.m_bar%self.r_bar
        return m_mod_r==1 or m_mod_r==(self.r_bar-1)
        
    def create_buckets(self,data):
        buckets=[Bucket(i) for i in range(self.r)]
        for element in data:
            h1=element.location*self.M1
            buckets[self.point_to_index(h1,self.r_bar)].append(element)
        buckets.sort(reverse=True)#sort buckets in descending order of size
        return buckets

    def jiggle_offsets(self,H_hat,H_b_hat,phi_hat,bucket):
        #start at a random point
        start_offset=self.generator.integers(0,self.m)

        found=False
        found_offset=np.ones((self.d,),dtype=self.int_type)
        #try every possible offset
        for i in range(self.r):
            #wrap around by the size of the hash table, then convert to a multi index
            phi_offset=self.index_to_point((start_offset+i)%self.m,self.m_bar)
            collision=False
            #now check every element in the bucket for collisions
            for element in bucket:
                h0=element.location*self.M0
                h1=element.location*self.M1
                index=self.point_to_index(h1,self.r_bar)
                #apply existing offset (phi_hat) everywhere except the current index
                if index==bucket.phi_index:
                    offset=phi_offset
                else:
                    offset=phi_hat[index]
                hash=h0+offset
                #check if the resulting hash is in use
                collision=H_b_hat[self.point_to_index(hash,self.m_bar)]
                if collision:
                    break
            if not collision:
                found=True
                found_offset=phi_offset
        if found:
            #update the offset table array with the new offset
            phi_hat[bucket.phi_index]=found_offset
            #and insert the bucket into the hash table at that location
            self.insert(bucket,H_hat,H_b_hat,phi_hat)
            return True
        else:
            return False

    def insert(self,bucket,H_hat,H_b_hat,phi_hat):
        for element in bucket:
            hashed=self.hash(element.location,phi_hat)
            i=self.point_to_index(hashed,self.m_bar)
            H_hat[i]=PerfectSpatialHashEntry(element,self.M2)
            H_b_hat[i]=True

    def hash(self,point,offset_table=None):
        '''
        return the index in the hash table for a given position in the domain

        Parameters: point : (d,) np integer array
                        location in the domain
                    offset_table : (self.r_bar,d) np integer array or None (default)
                        if None, uses self.phi
        Returns:    hash_position: integer
                        index into hash table (which is stored at self.H)
        '''
        if offset_table is None:
            offset_table=self.phi
        h0=point*self.M0
        h1=point*self.M1
        i=self.point_to_index(h1,self.r_bar)
        offset=offset_table[i]
        return h0+offset

    def hash_positions(self,data,H_hat):
        domain_size=self.u_bar**self.d
        indices=np.full(self.m,False)
        data_b=np.full(domain_size,False)
        for element in data:
            data_b[self.point_to_index(element.location,self.u_bar)]=True
        #sweep thru all points in domain without a data entry
        for i in range(domain_size):
            if data_b[i]:
                continue
            p=self.index_to_point(i,self.u_bar)
            l=self.point_to_index(self.hash(p),self.m_bar)
            #record if position hash collides with existing element
            indices[l]=H_hat[l].hk==entry_hash(p,self.M2,1)
        #go through stored collisions and record all points that map to those indices
        collisions=defaultdict(list)
        for i in range(domain_size):
            p=self.index_to_point(i,self.u_bar)
            l=self.point_to_index(self.hash(p),self.m_bar)
            if indices[l]:
                collisions[l].append(i)
        #try to change positional hash parameter until no collisions
        success=True
        for domain_index,colliding_indices in collisions:
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
        H_entry.increment_k(self.M2)
        #if k is 0, all values have been tried
        if H_entry.k==0:
            return False
        success=True
        for i in colliding_indices:
            #i is a domain index
            p=self.index_to_point(i,self.u_bar)
            hk=entry_hash(p, self.M2, H_entry.k)
            if np.any(H_entry.location!=p) and H_entry.hk==hk:
                success=False
                break
        #if the k didin't work, recurse
        if not success:
            return self.fix_k(H_entry,colliding_indices)
        return True

    def point_to_index(self,point,width):
        return np.ravel_multi_index(point,(width,)*self.d,mode="wrap")
    def index_to_point(self,index,width):
        return np.array(np.unravel_index(index,(width,)*self.d))

    def __getitem__(self,point):
        #find where element would be located
        i=self.point_to_index(self.hash(point),self.m_bar)
        #and check has correct positional hash
        if self.H[i].equals(point,self.M2):
            return self.H[i].contents
        else:
            raise ValueError("Element not found in map")
    def __setitem__(self,point,contents):
        i=self.point_to_index(self.hash(point),self.m_bar)
        if self.H[i].hk==1:
            #nothing exists at this location
            self.H[i]=PerfectSpatialHashEntry(data_tuple(point,contents),self.M2)
        elif self.H[i].equals(point,self.M2):
            #overwrite contents
            self.H[i].update(point,contents,self.M2)
        else:
            #failure
            raise ValueError("Unable to add element to map")

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
    return np.dot(point,(hash_parameter**np.arange(1,len(point)+1,dtype=point.dtype)))*prime
class Bucket(list):
    def __init__(self,offset_index):
        self.phi_index=offset_index
        super().__init__()
    def __lt__(self,rhs):
        return len(self)<len(rhs)

class PerfectSpatialHashEntry:
    def __init__(self,data_element=None,prime=None):
        self.k=1
        if data_element is not None:
            self.update(data_element.location,data_element.contents,prime)
        else:
            self.contents=None
            self.location=None
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