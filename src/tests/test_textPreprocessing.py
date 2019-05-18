# python integrated
import unittest

# dependencies
import numpy

# project
import utils.textPreprocessing as textPreprocessing


class TestSum(unittest.TestCase):

    def test_lowercasing(self):
        original = ['Hello', 'HELLO']
        result_intelligent = ['hello', 'HELLO']
        result_normal = ['hello', 'hello']

        # normal mode
        result = textPreprocessing.lowercasing(original)
        self.assertEqual(result[0], result_normal[0], (result[0]) + " should be " + (result_normal[0]))
        self.assertEqual(result[1], result_normal[1], str(result[1]) + " should be " + str(result_normal[1]))        

        # intelligent mode
        result = textPreprocessing.lowercasing(original, intelligent=True)
        self.assertEqual(result[0], result_intelligent[0], (result[0]) + " should be " + (result_intelligent[0]))
        self.assertEqual(result[1], result_intelligent[1], str(result[1]) + " should be " + str(result_intelligent[1]))
        
    def test_stemming(self):
        original = ["TROUBLE", "TROUBling", "troubled", "troubles"]
        expected = ["TROUBL", "troubl", "troubl", "troubl"]
        result = textPreprocessing.stemming(original)

        for position in range(len(original)):
            self.assertEqual(result[position], expected[position], str(result[position]) + " should be " + str(expected[position]))

    def test_noise_removal(self):
        original = "HELLO, I'm very happy !123"
        expected = ['HELLO', 'I', 'm', 'very', 'happy']
        result = textPreprocessing.noise_removal(original)

        for position in range(len(expected)):
            self.assertEqual(result[position], expected[position], str(result[position]) + " should be " + str(expected[position]))

    def test_stopWord_removal(self):
        #TODO: find out why nltk is keeping 'I'
        original = ['HELLO', 'I', 'm', 'very', 'happy']
        expected = ['HELLO', 'happy']
        result = textPreprocessing.stopWord_removal(original)

        for position in range(len(expected)):
            self.assertEqual(result[position], expected[position], str(result[position]) + " should be " + str(expected[position]))

if __name__ == '__main__':
    unittest.main()
