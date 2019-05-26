import numpy as np 

class placeholder:
    link={}
    link["pre"]=[]
    link["next"]=[]
    result=None
    shape={}
    def __init__(self,shape):
        self.shape["in"]=shape

class add:
    link={}
    link["pre"]=[]
    link["next"]=[]
    tensor={}
    tensor["out"]=None
    shape={}
    def __init__(self,prenode):
        self.shape["out"]=self.shape["in"]=prenode[0].shape
        for node in prenode:
            if self.shape != node.shape:
                raise(ValueError)
        self.link["pre"]=prenode
        for node in prenode:
            prenode.link["next"].append(self)
    def flow(self):
        sum_of_tensor=np.zeros(shape=self.shape["in"])
        for pre in self.link["pre"]:
            pre.result
                