import numpy as np
import timeit
import sys
import psh

def game_of_life_test(seed):
    rng=np.random.default_rng(seed)
    width=48
    data=[]
    data_b=np.full(width*width,False)
    maxint=np.iinfo(np.array([width]).dtype).max
    shape=np.array((width,)*2)
    for x in range(width):
        for y in range(width):
            if rng.uniform()<(1/20):
                element=psh.data_tuple(np.array([x,y]),True)
                data.append(element)
                data_b[psh.point_to_index(element.location,shape)]=True
    print("data size: "+str(len(data)))
    print("data density: "+str(len(data)/width**2))
    start=timeit.default_timer()
    hashmap=psh.create_psh(data,2,width,seed)
    stop=timeit.default_timer()
    try:
        element_size=sys.getsizeof(element)+sys.getsizeof(element.location)+sys.getsizeof(element.contents)
        content_size=sys.getsizeof(element)+sys.getsizeof(element.contents)
        original_data_size=width**3*(element_size)
    except NameError:
        element_size=0
        content_size=0
        original_data_size=0#no elements created
    print("original data: "+str(original_data_size/(1024**2))+" mb")
    hashmapsize=hashmap.memory_size()
    print("class size: "+str(hashmapsize/(1024**2))+" mb")
    print("compression factor vs dense: "+str(hashmapsize/width**3/sys.getsizeof(bool)))
    print("compression factor vs sparse: "+str(hashmapsize/(sys.getsizeof(data)+len(data)*element_size)))
    print("compression factor vs optimal: "+str(hashmapsize/(sys.getsizeof(data)+len(data)*content_size)))
    print("map creation time: ")
    print(str(stop-start)+" seconds")

    print("Exhaustive Test")
    missing=[]
    imaginary=[]
    for i in range(width**2):
        p=psh.index_to_point(i,shape)
        exists=data_b[i]
        try:
            hashmap[p]
            if not exists:
                imaginary.append(p)
        except KeyError:
            if exists:
                missing.append(p)
    print("missing ("+str(len(missing))+"): ")
    print(missing)
    print("Extra ("+str(len(imaginary))+"): ")
    print(imaginary)
    return hashmap,data

if __name__=="__main__":
    game_of_life_test(0)