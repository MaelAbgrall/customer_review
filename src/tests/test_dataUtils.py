# python integrated
import unittest

# dependencies
import numpy

# project
import utils.dataUtils as dataUtils


class TestSum(unittest.TestCase):

    def test_three_split(self):
        array = numpy.array([0, 1, 2, 3,])
        split = 0.5
        result = (numpy.array([0, 1]), numpy.array([2, 3]))
        output = dataUtils.three_split(array, split)
        self.assertTrue(( output[0] == result[0]).all(), str(output[0]) + " Should be " + str(result[0]))
        self.assertTrue(( output[1] == result[1]).all(), str(output[1]) + " Should be " + str(result[1]))

    #TODO: tests for clean text


if __name__ == '__main__':
    unittest.main()
