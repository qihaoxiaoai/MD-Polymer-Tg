# -*- coding: utf-8 -*-

#%%
import numpy as np
import random
from MDAnalysis.core.universe import Universe
from scipy.spatial.transform import Rotation
from numpy.linalg import norm
from doepy import build

#import MDAnalysis
#MDAnalysis.version.__version__
#%%
n = 60#00

u = Universe(r"data")
filename   = r"data"
atoms      = len(u.atoms)              # atoms numbers in each chain
total_mass = sum(u.atoms.masses) * n   # total mass

lx          = 200
ly          = 200
lz          = 200

lo = 0.2
hi = 0.8
print('{:.2}'.format(2.0))
print('Total atom number is {}'.format(len(u.atoms)*n))

#%%
def extract_atoms(filename):
    with open(filename, 'r') as file:
        atom = []
        for line in file:
            if "Atoms" in line:
                break
        for line in file:
            if "Bonds" in line:
                break
            data = line.strip().split()
            atom.append(data)
        return atom
def extract_bond(filename):
    with open(filename, 'r') as file:
        bonds = []
        for line in file:
            if "Bonds" in line:
                break
        for line in file:
            if "Angles" in line:
                break
            data = line.strip().split()
            bonds.append(data)
        return bonds
def extract_angle(filename):
    with open(filename, 'r') as file:
        angles = []
        for line in file:
            if "Angles" in line:
                break
        for line in file:
            if "Dihedrals" in line:
                break
            data = line.strip().split()
            angles.append(data)
        return angles
def extract_dihedral(filename):
    with open(filename, 'r') as file:
        dihedrals = []
        for line in file:
            if "Dihedrals" in line:
                break
        for line in file:
            if "Impropers" in line:
                break
            data = line.strip().split()
            dihedrals.append(data)
        return dihedrals
def extract_improper(filename):
    with open(filename, 'r') as file:
        impropers = []
        for line in file:
            if "Impropers" in line:
                break
        for line in file:
            if line.strip() == "\n":
                break
            data = line.strip().split()
            impropers.append(data)
        return impropers

#%%
pos_target = u.atoms.positions
    
#%%
data = build.maximin(
{'x':[lx*lo, lx*hi],
 'y':[ly*lo, ly*hi],
 'z':[lz*lo, lz*hi],},
num_samples = n)

x = data.iloc[:,0].values
y = data.iloc[:,1].values
z = data.iloc[:,2].values
randomseeds = np.vstack((x, y, z)).T[:n] 

#%%
mol      = []
atomtype = []

bondtype = []
bonda    = []
bondb    = []

angletype = []
anglea    = []
angleb    = []
anglec    = []

dihedraltype = []
dihedrala    = []
dihedralb    = []
dihedralc    = []
dihedrald    = []

impropertype = []
impropera    = []
improperb    = []
improperc    = []
improperd    = []

pos = np.zeros((n,len(pos_target),3))

atom     = [[float(x) for x in row] for row in extract_atoms(filename)] 
atom     = [matrix for matrix in atom if matrix]  
bonds     = [[int(x) for x in row] for row in extract_bond(filename)] 
bonds     = [matrix for matrix in bonds if matrix]  
angles    = [[int(x) for x in row] for row in extract_angle(filename)]  
angles    = [matrix for matrix in angles if matrix]    
dihedrals = [[int(x) for x in row] for row in extract_dihedral(filename)]  
dihedrals = [matrix for matrix in dihedrals if matrix]     
impropers = [[int(x) for x in row] for row in extract_improper(filename)]  
impropers = [matrix for matrix in impropers if matrix]     

for m in range (n):
    #print(m)
    atomtype.append(np.array(atom)[:,2])   
    str(atomtype).replace("'", "")
    mol.append(np.array(atom)[:,1]+m)   
    str(mol).replace("'", "")
    
    bondtype.append(np.array(bonds)[:,1])      
    bonda.append(np.array(bonds)[:,2] + len(u.atoms)*m)      
    bondb.append(np.array(bonds)[:,3] + len(u.atoms)*m)
    
    angletype.append(np.array(angles)[:,1])      
    anglea.append(np.array(angles)[:,2] + len(u.atoms)*m)      
    angleb.append(np.array(angles)[:,3] + len(u.atoms)*m)
    anglec.append(np.array(angles)[:,4] + len(u.atoms)*m)
    
    dihedraltype.append(np.array(dihedrals)[:,1])      
    dihedrala.append(np.array(dihedrals)[:,2] + len(u.atoms)*m)      
    dihedralb.append(np.array(dihedrals)[:,3] + len(u.atoms)*m)      
    dihedralc.append(np.array(dihedrals)[:,4] + len(u.atoms)*m)      
    dihedrald.append(np.array(dihedrals)[:,5] + len(u.atoms)*m)   
    
    impropertype.append(np.array(impropers)[:,1])      
    impropera.append(np.array(impropers)[:,2] + len(u.atoms)*m)      
    improperb.append(np.array(impropers)[:,3] + len(u.atoms)*m)      
    improperc.append(np.array(impropers)[:,4] + len(u.atoms)*m)      
    improperd.append(np.array(impropers)[:,5] + len(u.atoms)*m)   
    
#%%
pos_target = u.atoms.positions - [max(u.atoms.positions[:,0]),max(u.atoms.positions[:,1]),max(u.atoms.positions[:,2])]
for ss in range (n):
    print(ss+1)
    axis  = np.random.randint(-15,15,(1,3)).tolist() 
    axis  = axis / norm(axis)  # normalize the rotation vector first
    theta = random.uniform(0,2*np.pi);
    initial_x = randomseeds[ss][0]
    initial_y = randomseeds[ss][1]
    initial_z = randomseeds[ss][2]
    for j in range (len(pos_target)):
        rot = Rotation.from_rotvec(theta * axis)
        pos[ss][j][0] = rot.apply(pos_target)[j][0] + initial_x 
        pos[ss][j][1] = rot.apply(pos_target)[j][1] + initial_y 
        pos[ss][j][2] = rot.apply(pos_target)[j][2] + initial_z
    
#%% ---------------------- Write LAMMPS data files ---------------#
with open(r"chains.data".format(n),'w')as LAMMPS:
    # First line is a comment line 
    LAMMPS.write('The random Nylon66 system from Python\n\n')
    #----------------Header Line----------------#
    LAMMPS.write('{} atoms\n'.format(len(pos[0])*n))
    LAMMPS.write('{} bonds\n'.format(len(u.bonds)*n))
    LAMMPS.write('{} angles\n'.format(len(u.angles)*n))
    LAMMPS.write('{} dihedrals\n'.format(len(u.dihedrals)*n))
    LAMMPS.write('{} impropers\n\n'.format(len(u.impropers)*n))
    #-----------------Types defination--------------#
    LAMMPS.write('{} atom types\n'.format(7))
    LAMMPS.write('{} bond types\n'.format(max(bondtype[0])))
    LAMMPS.write('{} angle types\n'.format(max(angletype[0])))
    LAMMPS.write('{} dihedral types\n'.format(max(dihedraltype[0])))
    LAMMPS.write('{} improper types\n\n'.format(max(impropertype[0])))
    #--------------Specify Masses and dimensions------------------#
    LAMMPS.write('{} {} xlo xhi\n'.format(0, lx))
    LAMMPS.write('{} {} ylo yhi\n'.format(0, ly))
    LAMMPS.write('{} {} zlo zhi\n'.format(0, lz))
    #-------------- Specify Atoms information ------------------#
    LAMMPS.write('\n Masses \n\n')
    LAMMPS.write('{} {} # n3\n'.format(1, 14.006700))
    LAMMPS.write('{} {} # hn\n'.format(2, 1.007970))
    LAMMPS.write('{} {} # c2\n'.format(3, 12.011150))
    LAMMPS.write('{} {} # h\n'.format(4, 1.007970))
    LAMMPS.write('{} {} # n\n'.format(5, 14.006700))
    LAMMPS.write('{} {} # c''\n'.format(6, 12.011150))
    LAMMPS.write('{} {} # o''\n'.format(7, 15.999400))
    # Atoms section
    LAMMPS.write('\n Atoms  # full \n\n')
    # Atom_style: full----atom-Id; molecule-ID; atom-type; q; x; y; z;
	# Write Atoms Section
    id = 1
    for i in range (len(pos)):
        for j in range(len(pos[i])):            
            LAMMPS.write('{} {} {} {} {:.1f} {:.1f} {:.1f} {} {} {}\n'.format(id, i+1, int(atomtype[i][j]), 0,
                                                                              pos[i][j][0],
                                                                              pos[i][j][1],
                                                                              pos[i][j][2],0,0,0))  
            id = id + 1

    # Write Bonds Section
    LAMMPS.write('\nBonds   \n\n')
    id = 1
    for i in range(len(bonda)):
        for j in range(len(bonda[0])):
            LAMMPS.write('{} {} {} {} \n'.format(id, bondtype[i][j], bonda[i][j], bondb[i][j]))
            id = id + 1

    # Write Angles Section
    LAMMPS.write('\nAngles   \n\n')
    id = 1
    for i in range(len(anglea)):
        for j in range(len(anglea[0])): 
            LAMMPS.write('{} {} {} {} {}\n'.format(id, angletype[i][j], anglea[i][j], angleb[i][j], anglec[i][j]))
            id = id + 1

    # Write Dihedrals Section
    LAMMPS.write('\nDihedrals   \n\n')
    id = 1
    for i in range(len(dihedrala)):
        for j in range(len(dihedrala[0])):
            LAMMPS.write('{} {} {} {} {} {}\n'.format(id, dihedraltype[i][j], dihedrala[i][j], dihedralb[i][j], dihedralc[i][j], dihedrald[i][j]))
            id = id + 1
 
    # Write Impropers Section
    LAMMPS.write('\nImpropers   \n\n')
    id = 1
    for i in range(len(impropera)):
        for j in range(len(impropera[0])):
            LAMMPS.write('{} {} {} {} {} {}\n'.format(id, impropertype[i][j], impropera[i][j], improperb[i][j], improperc[i][j], improperd[i][j]))
            id = id + 1
                        