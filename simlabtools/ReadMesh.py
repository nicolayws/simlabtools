#Function for reading the mesh objects in .eco files produced by eCorr. Code written in March 2017 by Egil Fagerholt, Jacobo Díaz and Miguel Costas.

import sys
import struct
import numpy as np


class Mesh(object):
    """Reads eCorr meshes (*.ecm and *.eco) of version 1.8 or 1.9

    Usage:

    Mesh(filepath)
    """

    def __init__(self, filepath):
        self.filepath = filepath            # Path of the *.ecm or *.eco file to read
        self.filetype = None
        self.fileversion = None
        self.noIdOffset = 1                 # Offset for the node Id
        self.noElm = None                   # Number of elements in mesh
        self.noNod = None                   # Number of nodes in mesh
        self.noDupNodes = None              # Number of duplicate nodes in mesh (only used for node splitting)
        self.maxNodesPerElm = None          # Max number of nodes per element
        self.includeTargetVals = None       # Defines if target coordinates (3D) is included, 0 = No, 1 = Yes
        self.includeProjectionError = None  # Defines if projection error (3D) is included, 0 = No, 1 = Yes
        self.includeRefine = None           # Defines if mesh has refined parts (3D) is included, 0 = No, 1 = Yes
        self.includeParts = None            # Defines if mesh has parts (3D) is included, 0 = No, 1 = Yes
        self.includeVelocity = None         # Defines if velocity is included, 0 = No, 1 = Yes
        self.includeVelocityX = None        # Defines if velocity is included, 0 = No, 1 = Yes
        self.elm = None                     # Element connectivity matrix [noElm x maxNodesPerElm]
        self.actElms = None                 # Active elements array [noElm x 1]
        self.elmType = None                 # Element type array [noElm x 1], 0 => Q4
        self.elmPart = None                 # Optional, Defined which part an element is part of [noElm, 1]
        self.nloc = None                    # Initial nodal locations in pixels [noNod, 2]
        self.ndef = None                    # Nodal displacements in pixels [noNod, 2]
        self.nlocRef = None                 # Nodal displacementsd at current reference [noNod, 2]
        self.nvel = None                    # Optional. Nodal velocities in pixels [noNod, 2]
        self.dupNodes = None                # Optional. Duplicate nodes [noDupNodes, 2], Used for node splitting
        self.dupNodesFixed = None           # Optional. Defines which duplicate nodes are fixed/released [noDupNodes, 1], Used for node splitting
        self.nlocX = None                   # Only 3D-DIC. Initial nodal locations in target coordinates [mm] [noNod, 3]
        self.ndefX = None                   # Only 3D-DIC. Nodal displacements in target coordinates [mm] [noNod, 3]
        self.nvelX = None                   # Optional, only for 3D-DIC. Nodal velocities in target coordinates [mm] [noNod, 3]
        self.targetProjectionError = None   # Optional, only for 3D-DIC. Projection error for camera models, [noNod, 2]

        self._eCorrReadMesh(self.filepath)
        self._postprocess()

    def __repr__(self):
        string = 'eCorr mesh\n' + 60*'-' + '\n'
        string += 'Elements: {0}\nNodes:{1}\n'.format(self.noElm, self.noNod)
        string += '\nMaximum nodal displacement X: {0} (Node {1})\n'.format(self.max_displ_x, self.max_displ_x_pos)
        string += 'Maximum nodal displacement Y: {0} (Node {1})\n'.format(self.max_displ_y, self.max_displ_y_pos)
        string += 'Minimum nodal displacement X: {0} (Node {1})\n'.format(self.min_displ_x, self.min_displ_x_pos)
        string += 'Minimum nodal displacement Y: {0} (Node {1})\n'.format(self.min_displ_y, self.min_displ_y_pos)
        string += '\nElement connectivity:\n{0}'.format(self.elm.__repr__())

        return string

    def _ReadBoolean(self, f):
        val = struct.unpack('b', f.read(1))[0]
        if val:
            val = True
        else:
            val = False
        return val

    def _ReadByte(self, f):
        return struct.unpack('b', f.read(1))[0]

    def _ReadInt(self, f):
        return struct.unpack('i', f.read(4))[0]

    def _ReadDouble(self, f):
        return struct.unpack('d', f.read(8))[0]

    def _eCorrReadMesh(self, filepath):
        try:
            f = open(filepath, "rb")

            self.filetype = self._ReadInt(f)
            if self.filetype != 0:
                raise Exception("Not an eCorr mesh file!")

            self.fileversion = self._ReadDouble(f)
            if self.fileversion != 1.9 and self.fileversion != 1.8 and self.fileversion != 2.3:
                raise Exception("Only implemented for eCorr mesh file version 1.8 or 1.9!")

            self.noElm = self._ReadInt(f)
            self.noNod = self._ReadInt(f)
            self.noDupNod = self._ReadInt(f)
            self.maxNodesPerElm = self._ReadInt(f)
            self.includeTargetVals = self._ReadInt(f)
            self.includeProjectionError = self._ReadInt(f)
            self.includeRefine = self._ReadInt(f)
            self.includeParts = self._ReadInt(f)
            for i in range(0, 10):
                dummy = self._ReadInt(f)

            # Element connectivity
            self.elm = np.zeros((self.noElm, self.maxNodesPerElm))
            for i in range(0, self.noElm):
                for j in range(self.maxNodesPerElm):
                    self.elm[i][j] = self._ReadInt(f) + self.noIdOffset

            # Active Elements
            self.actElms = np.zeros((self.noElm, 1))
            for i in range(self.noElm):
                self.actElms[i] = self._ReadBoolean(f)

            # Element Type
            self.elmType = np.zeros((self.noElm, 1))
            for i in range(self.noElm):
                self.elmType[i][0] = self._ReadByte(f)

            # Mesh parts
            if self.includeParts:
                self.elmPart = np.zeros((self.noElm, 1))
                for i in range(self.noElm):
                    self.elmPart[i][0] = self._ReadByte(f)

            # Nodal initial locations
            self.nloc = np.zeros((self.noNod, 2))
            for i in range(self.noNod):
                self.nloc[i][0] = self._ReadDouble(f)
                self.nloc[i][1] = self._ReadDouble(f)

            # Nodal displacements
            self.ndef = np.zeros((self.noNod, 2))
            for i in range(self.noNod):
                self.ndef[i][0] = self._ReadDouble(f)
                self.ndef[i][1] = self._ReadDouble(f)

            # Nodal reference locations
            self.nlocRef = np.zeros((self.noNod, 2))
            for i in range(self.noNod):
                self.nlocRef[i][0] = self._ReadDouble(f)
                self.nlocRef[i][1] = self._ReadDouble(f)

            if self.fileversion > 1.8:
                # Nodal velocity (if exists)
                self.includeVelocity = self._ReadBoolean(f)
                if self.includeVelocity:
                    self.nvel = np.zeros((self.noNod, 2))
                    for i in range(self.noNod):
                        self.nvel[i][0] = self._ReadDouble(f)
                        self.nvel[i][1] = self._ReadDouble(f)

            # Duplicate nodes
            self.dupNodes = np.zeros((self.noDupNod, 2))
            self.dupNodesFixed = np.zeros((self.noDupNod, 1))
            for i in range(self.noDupNod):
                self.dupNodes[i][0] = self._ReadInt(f)
                self.dupNodes[i][1] = self._ReadInt(f)
                self.dupNodesFixed[i][1] = self._ReadBoolean(f)

            # Target coordinates (3D)
            if self.includeTargetVals:
                # Nodal initial target locations (3D)
                self.nlocX = np.zeros((self.noNod, 3))
                for i in range(self.noNod):
                    self.nlocX[i][0] = self._ReadDouble(f)
                    self.nlocX[i][1] = self._ReadDouble(f)
                    self.nlocX[i][2] = self._ReadDouble(f)

                # Nodal target displacements (3D)
                self.ndefX = np.zeros((self.noNod, 3))
                for i in range(self.noNod):
                    self.ndefX[i][0] = self._ReadDouble(f)
                    self.ndefX[i][1] = self._ReadDouble(f)
                    self.ndefX[i][2] = self._ReadDouble(f)

                if self.includeProjectionError:
                    self.targetProjectionError = np.zeros((self.noNod, 2))
                    for i in range(self.noNod):
                        self.targetProjectionError[i][0] = self._ReadDouble(f)
                        self.targetProjectionError[i][1] = self._ReadDouble(f)

                if self.fileversion > 1.8:
                    # Nodal velocity (if exists)
                    self.includeVelocityX = self._ReadBoolean(f)
                    if self.includeVelocityX:
                        self.nvelX = np.zeros((self.noNod, 3))
                        for i in range(self.noNod):
                            self.nvelX[i][0] = self._ReadDouble(f)
                            self.nvelX[i][1] = self._ReadDouble(f)
                            self.nvelX[i][2] = self._ReadDouble(f)

                # Refined mesh (Not properly implemented)
                if self.includeRefine:
                    noRefine = self._ReadInt(f)
                    noRefinePerElm = self._ReadInt(f)
                    for i in range(self.noElm):
                        for j in range(noRefinePerElm):
                            dummy = self._ReadInt(f)
                    for i in range(noRefine):
                        for j in range(0, 1):
                            dummy = self._ReadDouble(f)
                            dummy = self._ReadDouble(f)
                            dummy = self._ReadDouble(f)

            f.close()

        except IOError as err:
            print("eCorrReadMesh error: {0}".format(err))

        except Exception as err:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("eCorrReadMesh error in line {0}: {1}".format(exc_traceback.tb_lineno, err))
            f.close()

    def _postprocess(self):
        self.max_displ_x = self.ndef[:, 0].max()
        self.max_displ_x_pos = self.ndef[:, 0].argmax() + self.noIdOffset
        self.max_displ_y = self.ndef[:, 1].max()
        self.max_displ_y_pos = self.ndef[:, 1].argmax() + self.noIdOffset

        self.min_displ_x = self.ndef[:, 0].min()
        self.min_displ_x_pos = self.ndef[:, 0].argmin() + self.noIdOffset
        self.min_displ_y = self.ndef[:, 1].min()
        self.min_displ_y_pos = self.ndef[:, 1].argmin() + self.noIdOffset
