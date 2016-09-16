# -*- coding: utf-8 -*-
"""
Created on Tue Jun 07 10:33:12 2016

@author: Carprees
"""

### Función para conocer el tipo de una variable. ###
# param @in var: cualquier tipo de variable .
def varType(var):
    print(type(var))
  
### Función para deshacer listas de listas. ###
# param @in var: lista de listas.
def unlist(var):
    try:
        resul=[]
        for a in var:
            resul+=a
        return resul
    except:
        print('unlist Error: La variable de entrada debe ser una lista de listas.')
    
### Para comprimir un archivo. ###
# param @in name: nombre completo del archivo a comprimir.
def czip(name):
    import shutil
    import gzip
    with open(str(name), 'rb') as f_in, gzip.open(str(name)+'.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        
### Función que dumpea un objeto con una estructura en un fchero que depués comprime. ###
# param @in obj: Objeto a dumpear con pickle.
# param @in args1: Variable utilizada como entrada para un str con el nombre 
#                  deseado para el fichero o un boolean para confirmar borrado
#                  del fichero original.
# param @in args2: Idem a anterior.
        
def cPickleZip(obj, args1=None, args2=None):
    import six.moves.cPickle as pickle
    import shutil
    import gzip
    import os
    
    name = 'save.car'
    rmOriginal = True
    
    if (isinstance(args1,str)):
        name = args1
    elif (isinstance(args1,bool)):
        rmOriginal = args1
        
    if (isinstance(args2,str)):
        name = args2
    elif (isinstance(args2,bool)):
        rmOriginal = args2
    
    pickle.dump( obj, open( name, "wb" ) )
    
    with open(name, 'rb') as f_in, gzip.open(name+'.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    if rmOriginal:
        os.remove(name)
 
def saveImage(data, name='outfile.jpg'):   
    import scipy.misc
    scipy.misc.imsave(name, data)    
        