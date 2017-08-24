#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import mrmProcessing.model.sample as sample
import mrmProcessing.model.batch as batch
import mrmProcessing.model.project as project

class testSample(unittest.TestCase):
    proj=project.project("projID")
    bat=batch.batch("batchID",proj)
    def test_001_init(self):
        """Tests if not having any parameters in the init call raises a TypeError"""
        self.assertRaises(TypeError,sample.sample)
    def test_002_init(self):
        """Tests if having only id parameter in the init call raises a TypeError"""
        self.assertRaises(TypeError,sample.sample,"id")
    def test_003_init(self):
        """Tests if having id and batch parameter in the init call raises a TypeError"""
        self.assertRaises(TypeError,sample.sample,"id",testSample.bat)
    def test_004_init(self):
        """Tests if having id, batch and filename parameter in the init call raises a TypeError"""
        self.assertRaises(TypeError,sample.sample,"id",testSample.bat,"file_name")
    def test_005_init(self):
        """Tests if passing the id, batch, filename and sample name parameters generates a new object of type sample"""
        self.assertIsInstance(sample.sample("id",testSample.bat,"file_name","sample_name"),sample.sample)

if __name__ == '__main__':
    unittest.main()