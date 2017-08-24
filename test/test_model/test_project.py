#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import mrmProcessing.model.project as project


class testProject(unittest.TestCase):
    def test_001_init(self):
        """Tests if not having any parameters in the init call raises a TypeError"""
        self.assertRaises(TypeError,project.project)
    def test_002_init(self):
        """Tests if having the id parameters generates a new object of type project"""
        self.assertIsInstance(project.project("id"),project.project)
    def test_003_init(self):
        """Tests if having the named parameter "name" generates a new object of type project"""
        self.assertIsInstance(project.project("id",name="name"),project.project)
    def test_004_init(self):
        """Tests if having the named parameter "basePath" generates a new object of type project"""
        self.assertIsInstance(project.project("id",basePath="basePath"),project.project)

if __name__ == '__main__':
    unittest.main()