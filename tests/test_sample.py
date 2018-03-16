#!/usr/bin/python
#test drawing a random sample whether the output dimensions are ok and if the outputfile is found

import unittest
import numpy as np
import os


from digitgenerator import Digitgenerator

#output test image name
outputname='./test_output.png'

#make a random digit list (5-10 digits long)
digitlistlen=np.random.randint(5,11)
digitlist=list(np.random.randint(10, size=digitlistlen))

#initialize a Digitgenerator instance
digitgen=Digitgenerator()
outputarray=digitgen.draw_from_mnist(digitlist,(1,2),0)


#check that the output dimenisons are ok
class GeneratorTestCase1(unittest.TestCase):
    def test_outputdims(self):
    
        #is the first dimension 28
        self.assertTrue(np.shape(outputarray)[0]==28)
        
        #width is ok?
        self.assertTrue(np.shape(outputarray)[1]>28*digitlistlen)
        self.assertTrue(np.shape(outputarray)[1]<(28*digitlistlen+1))
        
#check that the output file is found
class GeneratorTestCase1(unittest.TestCase):
    def test_outputdims(self):
    
        digitgen.save_latest_as_png(filename=outputname)
        
        #the saved file exists?
        self.assertTrue(os.path.isfile(outputname))
    
        print('test output image '+outputname+' generated from test sample',digitlist)

   

if __name__ == "__main__":
    unittest.main()
