#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:14:09 2021

@author: ubuntu
"""
from ast import Interactive
import meshio
import dolfin
from dolfin import *    
#import mshr 
import os
import matplotlib.pyplot as plt
import numpy as np
import time

#%%
def solve_poroelasticity( output_dir,model,per,KKresortee,ii,C_bir2019,Beta_bir2019,C1_bir2019,C3_bir2019):
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        if prune_z:
            out_mesh.prune_z_0()
        return out_mesh
        
    msh=meshio.read("octavo_de_esfera.msh")


    triangle_mesh = create_mesh(msh, "triangle", False)
    tetra_mesh = create_mesh(msh, "tetra", False)
    meshio.write("mesh.xdmf", tetra_mesh)
    meshio.write("mf.xdmf", triangle_mesh) 
    #print(msh.cell_data_dict)
    #from dolfin import *
    parameters["allow_extrapolation"] = True 
    parameters["form_compiler"]["optimize"] = True 

    mesh=dolfin.Mesh()
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with XDMFFile("mesh.xdmf") as infile:
        infile.read(mesh)
        infile.read(mvc, "name_to_read")
    cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)


    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
    with XDMFFile("mf.xdmf") as infile:
        infile.read(mvc, "name_to_read")   
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 3

    gdim = mesh.geometry().dim()
  
    ds = Measure("ds", domain=mesh, subdomain_data=mf)

    dx = Measure("dx", domain=mesh, subdomain_data=cf)

    # Limit quadrature degree
    dx = dx(degree=3)
    ds = ds(degree=3)

    # Build function space
    V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)  ##
    # Definicion de espacios para formulacion mixta
    P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1  #elemento mixto
    W = dolfin.FunctionSpace(mesh, TH)  #creo el nuevo espacio de funciones (mixto) DESPLAZAMIENTO Y PRESION
    info("Num DOFs {}".format(W.dim()))

    empotrado= dolfin.Constant((0.0, 0.0, 0.0))
    bc_emp=dolfin.DirichletBC(W.sub(0), empotrado, mf,3)     
    pressure = dolfin.Expression(("presiones"),degree=2,domain=mesh,presiones=0.0)
    bc_pinf= dolfin.DirichletBC(W.sub(1),  pressure, mf,2) #presion en bronquio inf
    # Define Dirichlet boundary
    zero = Constant(0.0)
    bc_sym1= dolfin.DirichletBC(W.sub(0).sub(0), zero, mf, 5)     # u.n = 0 left
    bc_sym2= dolfin.DirichletBC(W.sub(0).sub(1), zero, mf, 4)   # u.n = 0   bottom
    bc_sym3= dolfin.DirichletBC(W.sub(0).sub(2), zero, mf, 6)    # u.n = 0  front
    # Save subdomains
    file_sd = dolfin.File("boundaryes.pvd")
    file_sd << mf
    bcs=[bc_pinf, bc_sym1, bc_sym2, bc_sym3]
    
    print(ds)
    
    if model=='ber':
        def stress(u, p):
            E, nu = 0.73, 0.3 #berger full
            mu, lmbda = dolfin.Constant(E/(2*(1 + nu))),dolfin.Constant(E*nu/((1 + nu)*(1 - 2*nu)))
            phi0=Constant(0.99)
            F = variable(I + grad(u))
            C=variable(F.T*F)
            J=sqrt(det(C))
            psi=0.5*mu*(tr(C)-3)+0.25*lmbda*(J*J-1)-(mu+0.5*lmbda)*ln(J-1+phi0)  
            S=2*diff(psi,C)
            PK=F*S  -p*J*inv(F).T 

            
            return PK
    
    elif model=='bel':
        def stress(u, p):

                F = (I + grad(u))
                C=variable(F.T*F)
                I1C=variable(tr(C))
                I2C=0.5*((tr(C)**2)-tr(C*C))
                I3C=variable(det(C))
                J=variable(sqrt(I3C))
                
                I1raya=I1C*pow(I3C,-1/3)
                k1=Constant(4.34)
                k2=Constant(5.92)
                kk=Constant(72.5)        
                
                psi=(k1/(2*k2))*(exp(k2*((I1raya/3)-1))-1)+(0.25*kk)*(-2*ln(J)+J*J-1)
                
                S=2*diff(psi,C)
                PK=F*S-p*J*inv(F).T 
                
                return PK 

    elif model=='rausch':
        def stress(u, p):

                F = (I + grad(u))
                C=variable(F.T*F)
                I1C=variable(tr(C))
                I2C=0.5*((tr(C)**2)-tr(C*C))
                I3C=variable(det(C))
                J=variable(sqrt(I3C))
                
                I1raya=I1C*pow(I3C,-1/3)
                k1=Constant(4.34)
                k2=Constant(5.92)
                kk=Constant(72.5)        
                
                cquad=Constant(4.1)
                ccub=Constant(20.7)
                kappa=Constant(16.5)
                
                psi=(cquad*(I1raya-3)**2)+(ccub*(I1raya-3)**3)+0.25*kappa*(-2*ln(J)+J*J-1)
                
                
                S=2*diff(psi,C)
                PK=F*S-p*J*inv(F).T 
                
                return PK 
            
    elif model=='bir2019':
        def stress(u, p):
                F = (I + grad(u))
                C=variable(F.T*F)
                I1C=variable(tr(C))
                I2C=0.5*((tr(C)**2)-tr(C*C))
                I3C=variable(det(C))
                J=variable(sqrt(I3C))
                
                I1raya=I1C*pow(I3C,-1/3)
                k1=Constant(4.34)
                k2=Constant(5.92)
                kk=Constant(72.5)        
                
                cquad=Constant(4.1)
                ccub=Constant(20.7)
                kappa=Constant(16.5)
                
                psi=(cquad*(I1raya-3)**2)+(ccub*(I1raya-3)**3)+0.25*kappa*(-2*ln(J)+J*J-1)

                c_bir2019 = Constant(C_bir2019)
                beta_bir2019 = Constant(Beta_bir2019)
                c1_bir2019 = Constant(C1_bir2019)
                c3_bir2019 = Constant(C3_bir2019)
                
                #psi=(356.7/1000)*(I1C-3)  +  (331.7/1000)*(pow(I3C,-1.075)-1)+(278.2/1000)*pow((pow(I3C,-1/3)*I1C-3),3)+(5.766/1000)*pow((pow(I3C,1/3)-1),6)
                psi=(c_bir2019/1000)*(I1C-3)  +  ((c_bir2019/beta_bir2019)/1000)*(pow(I3C,-beta_bir2019)-1)+((c1_bir2019)/1000)*pow((pow(I3C,-1/3)*I1C-3),3)+((c3_bir2019)/1000)*pow((pow(I3C,1/3)-1),6)
                
                S=2*diff(psi,C)
                PK=F*S-p*J*inv(F).T 
                
                return PK 
                       
            
    elif model=='yoshi':
        
        def stress(u, p):
            F = I + grad(u)  #yoshihara
            J = det(F)
            C=variable(F.T*F)
            I1C=tr(C)
            I2C=0.5*((tr(C)**2)-tr(C*C))
            I3C=det(C)
            cc=Constant(1.298)
            Beta=Constant(0.75)
            psi=cc*(I1C-3)+(cc/Beta)*((pow(I3C,-1*Beta)-1))
            S=2*diff(psi,C)
            PK=F*S-p*J*inv(F).T 
            return PK
    
    elif model=='ma':
        def stress(u, p):
                F = I + grad(u)
                J = det(F)
                C=variable(F.T*F)
                E=0.5*(C-I)
                I1E=tr(E)
                I2E=0.5*((tr(E)**2)-tr(E*E))
                I3E=det(E)
                cc=Constant(1)
                aa=Constant(1)
                bb=Constant(1)
              
                psi=cc*exp(aa*I1E*I1E+bb*I2E)
               
                #B = F * F.T
                #T = -p*I + mu*(B-I) # Cauchy stress
                #S = J*T*inv(F).T # 1st Piola-Kirchhoff stress
                S=2*diff(psi,C)
                #PK=F*S
                PK=F*S-p*J*inv(F).T 
                return PK
    
    elif model=='bir':
        def stress(u, p):
                F = I + grad(u)
                J = det(F)
                C=(F.T*F)
                I1C=tr(C)
                I3C=det(C)
                cc=Constant(286.61/1000)
                Beta=Constant(1.1738)
                cd=Constant(0.008238/1000)
                dd=Constant(6)        
                dif=cc*I+(cc/Beta)*(-Beta  *  (pow(I3C,(-Beta-1)))*I3C*inv(C)) + cd*dd*(pow((I1C-3),(dd-1)))*I
                S=2*dif
                PK=F*S-p*J*inv(F).T 
                
                return PK 
      
    def F(u,p):
        F = I + grad(u)
        return F

    def Finv(u,p):
        F = I + grad(u)
        Finverse=inv(F)
        return Finverse
    def Finv_t(u,p):
        F = I + grad(u)
        Finverse=inv(F)
        Finvt=Finverse.T
        return Finvt
    
    def JJ(u,p):
        F = I + grad(u)
        J=det(F)
        return J

    
    def lnJJ(u,p): ###OBSE
        F = I + grad(u)
        J=det(F)
        lnJJ=ln(J)
        return J
    
    def phi(u,p):
        F = I + grad(u)
        J=det(F)
        phi0=0.99
        phi=J-1+phi0
        return phi
       
    def F2aux_mass(u,p):
                
        F = I + grad(u)
        Finverse=inv(F)
        Finvt=Finverse.T
        J=det(F)
        KK=Constant(per)
        K=KK*I       
        aux=J*Finverse*K*Finvt* grad(p)

        return aux


    
    # Timestepping theta-method parameters
    dt = dolfin.Expression(("beta"), beta=0., degree=2, domain=mesh)
    # Unknowns, values at previous step and test functions
    w = Function(W)
    u, p = split(w)
    w0 = Function(W)
    

    
    u0, p0 = split(w0)

    _u, _p = TestFunctions(W)
    du = dolfin.TrialFunction(W)            # Incremental displacement , sirve para Jacobian solamente
   
    
    I = Identity(W.mesh().geometry().dim())

    # Balance of momentum
    P=stress(u,p)
    P0=stress(u0,p0)
    F1=dolfin.inner(P, dolfin.grad(_u) )*dolfin.dx #+dolfin.dot(T,_u)*dolfin.ds(subdomain_data=boundaries,subdomain_id=2)#+dolfin.dot(T,v)*dolfin.ds(1)

    F2aux1=JJ(u,p)*tr((1/1)*(grad(u)-grad(u0))*Finv(u,p)) #proy continuo
    lnJJn1=lnJJ(u,p)
    lnJJn=lnJJ(u0,p0)

    
    F2aux2=F2aux_mass(u,p)
    flujo=dolfin.Constant(-0.0000)

    F2=  (dolfin.inner(F2aux1, _p))*dolfin.dx+ dt*(dolfin.inner(grad(_p),F2aux2))*dolfin.dx # + dolfin.inner(_p,Constant(0))*ds(1)#+dt*(dolfin.inner(grad(_p),F2aux3))*dolfin.dx #+dt*(dolfin.inner(F2aux2,F2aux3))*dolfin.dx
    
    #F2=   dolfin.inner(grad(_p),F2aux2)*dolfin.dx #ESTATICO
    Kresorte=KKresortee
    n = FacetNormal(mesh)
    unormal=(dolfin.dot(u,n)*n)


    R3=dolfin.inner(u,_u)*Kresorte*dolfin.ds(subdomain_data=mf,subdomain_id=3)#+dolfin.inner(u,_u)*Kresorte*dolfin.ds(subdomain_data=boundary_markers,subdomain_id=1)#+dolfin.inner(u,_u)*Kresorte*dolfin.ds(subdomain_data=boundary_markers,subdomain_id=3)
    R4=dolfin.inner(u,_u)*(Kresorte)*dolfin.ds(subdomain_data=mf,subdomain_id=2)#+dolfin.inner(u,_u)*(Kresorte)*dolfin.ds(subdomain_data=boundary_markers,subdomain_id=4)
    R=F1+F2+R3+R4

    
    Jac = dolfin.derivative(R, w, du)
    
    
    # Initialize solver
    problem = NonlinearVariationalProblem(R, w, bcs=bcs, J=Jac,form_compiler_parameters={'optimize':True})
    solver = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.parameters['newton_solver']['maximum_iterations'] = 10
    solver.solve()

    # Extract solution components
    u, p = w.split()
    u.rename("u", "displacement")
    p.rename("p", "pressure")

    # Create files for storing solution
    ufile = XDMFFile(os.path.join(output_dir, "dispt.xdmf"))
    pfile = XDMFFile(os.path.join(output_dir, "prest.xdmf"))

    # Time-stepping loop
    t = 0
    if model=='ber':
        # Save solution in VTK format
        folder_name = "REsults/"
        filedisplacement = File(folder_name+"desplatt.pvd", "compressed")
        filepressure = File(folder_name+"presiontt.pvd", "compressed")
    
    valores_u=[]
    tiempos=[]
    valores_p=[]
    Jacob=[]
    fluxes=[]
    drop=[]
    
    
    
    t=0
    tiempos=[]
    presionestodas=[]
    ##############################

    Nciclos=2 ##debe ser 2
    p_min=0
    p_max=1.97*3/10
    for ciclo in np.arange(1,Nciclos+1):
        duration_step=3
        t0=duration_step*(ciclo-1)
        t1=0.1+t0
        t2=1+t0
        t3=1.025+t0
        t4=duration_step+t0
        t34=1.5+t0
    
        if ciclo<=Nciclos/2:
            factor=-1
        else:
            factor=-1
        p0=p_min
        p1=p_max
        p2=p_max
        p3=p_min
        p34=p_min
        p4=p_min
        
        n0=10 #subida
        n1=15 # flujo >0
        n2=10 #bajada
        n23=10
        n3=15 #nulo
        if ciclo==Nciclos:    
            end=True
        else:
            end=False
            
        times0=np.linspace(t0,t1,n0,endpoint=False)
        times1=np.linspace(t1,t2,n1,endpoint=False)
        times2=np.linspace(t2,t3,n2,endpoint=False)
        times3=np.linspace(t3,t34,n23,endpoint=False)
        times4=np.linspace(t34,t4,n3,endpoint=end)
        timesaux=np.concatenate((times0,times1,times2,times3,times4))
        
        ps0=np.linspace(p0,p1,n0,endpoint=False)
        ps1=np.linspace(p1,p2,n1,endpoint=False)
        ps2=np.linspace(p2,p3,n2,endpoint=False)
        ps3=np.linspace(p3,p34,n23,endpoint=False)
        ps4=np.linspace(p34,p4,n3,endpoint=end)
        psaux=np.concatenate((ps0,ps1,ps2,ps3,ps4))
        
        if ciclo==1:
            times=timesaux
            ps=psaux
        else:
            times=np.concatenate((times,timesaux))
            ps=np.concatenate((ps,psaux))
    times=times[1:]
    ps=ps[1:]
    
    dts=[]
    for i in np.arange(len(times)):
        if i==0:
            dts.append(times[i])
        else:
            dts.append(times[i]-times[i-1])

    for i in np.arange(len(times)):
        t=times[i]
        dtt=dts[i]
        presion=ps[i]

        dt.beta=dtt
        pressure.presiones=presion
        presionestodas.append(presion)
        print('PRESSURE,TIME,MODEL:',presion,t,model)   
        # Prepare to solve and solve
        w0.assign(w)
        solver.solve()
        # Store solution to files and plot
        ufile.write(u, t)
        pfile.write(p, t)
        
        dif=0
        drop.append(dif)

        if model=='ber':
            filedisplacement << (u, t) #FOR VTK
            filepressure << (p, t)
        u_array = u.vector()[:]

        
        u_array = u.vector().get_local()
        p_array = p.vector().get_local()
        

        valores_u.append(u_array)
        valores_p.append(p_array)
        FF = I + dolfin.grad(u)             # Deformation gradient      
        vol=dolfin.assemble((det(I+grad(u)))*dx)
        volL=1000*vol/(10**9)
        Jacob.append(volL)
        n = FacetNormal(mesh)
        deltavol=dolfin.assemble(((JJ(u,p)-JJ(u0,p0))/dt)*dx)
        fluxes.append(deltavol*10**-6)





        areas=Constant(1)*ds(1)
        area=dolfin.assemble(areas)
        
        np.save(model+str(ii)+'tiempos.npy',times[0:i+1])
        np.save(model+str(ii)+'fluxes.npy',fluxes)
        np.save(model+str(ii)+'presionestodas.npy',presionestodas)
        np.save(model+str(ii)+'volumenes.npy',Jacob)

    # Close files
    ufile.close()
    pfile.close()
    
    return times,Jacob,fluxes,presionestodas


from pyDOE import lhs

# we consider a range of 30% around the mean

# Constitutive model parameter bounds (Ma et al.)
range = 30/100
a = [-range*0.43 + 0.43 , 0.43 +range*0.43]
b = [+range*(-0.6) + (-0.6), (-0.6) - range*(-0.6)]
c = [-range*2 + 2 , 2 +range*2]

# Constitutive model parameter bound (Bir. 2019 et al.)
range = 50/100
C_bir2019    = [-range*356.7 + 356.7 , 356.7 +range*356.7]
Beta_bir2019 = [-range*1.075 + 1.075 , 1.075 +range*1.075]
C1_bir2019   = [-range*278.2 + 278.2 , 278.2 +range*278.2]
C3_bir2019   = [-range*5.766 + 5.766 , 5.766 +range*5.766]
                
# Mechanical parameters
per = [-range*10000 + 10000 , 10000 +range*10000]
KKresortee = [-range*0.08 + 0.08 , 0.08 +range*0.08]


# We use Latin Hypercube Sampling approach to generate space-filling training samples.
# Training data
sample_num = 200
#lb = np.array([a[0], b[0], c[0], per[0], KKresortee[0]])
#ub = np.array([a[1], b[1], c[1], per[1], KKresortee[1]])
lb = np.array([C_bir2019[0], Beta_bir2019[0], C1_bir2019[0], C3_bir2019[0], per[0], KKresortee[0]])
ub = np.array([C_bir2019[1], Beta_bir2019[1], C1_bir2019[1], C3_bir2019[1], per[1], KKresortee[1]])
#X_data = (ub-lb)*lhs(5, samples=sample_num) + lb
X_data = (ub-lb)*lhs(6, samples=sample_num) + lb

print(X_data)
# Visualizing training samples
import plotly.express as px

#fig = px.scatter_3d(X_train, x=X_train[:,0], y=X_train[:,1], z=X_train[:,2])
#fig.show()

Y_data = np.empty([sample_num, 2])
print(Y_data)

start_time = time.time()

i = 0

while i < len(X_data):

    ii = i
    #a = X_data[i,0] 
    #b = X_data[i,1]
    #c = X_data[i,2]
    C_bir2019    = X_data[i,0] 
    Beta_bir2019 = X_data[i,1]
    C1_bir2019   = X_data[i,2]
    C3_bir2019   = X_data[i,3]
    #per = X_data[i,3]
    #KKresortee = X_data[i,4]
    per          = X_data[i,4]
    KKresortee   = X_data[i,5]
    
    models=['bir2019']
    
    for model in models:
        tiempos,Jacob,flux,presionestodas=solve_poroelasticity( 'TEST',model,per,KKresortee,ii,C_bir2019,Beta_bir2019,C1_bir2019,C3_bir2019)

    from sklearn import linear_model

    #Debe existir una carpeta llamada derecho donde estén las curvas del pulmón derecho. Idem para el izquierdo
    name='bir2019'

    ##AMBOS
    fluxes=fileflujos=np.load(name+str(ii)+'fluxes.npy')*8+np.load(name+str(ii)+'fluxes.npy')*8#[0:84]
    presionestodas=filepresiones=np.load(name+str(ii)+'presionestodas.npy')#+filepresionesi
    tiempos=filetiempos=np.load(name+str(ii)+'tiempos.npy')
    Jacob=filevolumenes=np.load(name+str(ii)+'volumenes.npy')*8+np.load(name+str(ii)+'volumenes.npy')*8#[0:84]
    Jacob=Jacob-Jacob[0]

    #También en una carpeta llamada ambos (crear) se guardan los npy de ambos pulmones
    lado='./ambos/'
    model=name
    np.save(lado+model+str(ii)+'tiempos.npy',tiempos)
    np.save(lado+model+str(ii)+'fluxes.npy',fluxes)
    np.save(lado+model+str(ii)+'presionestodas.npy',presionestodas)
    np.save(lado+model+str(ii)+'volumenes.npy',Jacob)

    def regression(fileflujos,filepresiones,filetiempos,filevolumenes,name):
        flujos=np.asarray((fileflujos))*1/60
        presiones=10.2*np.asarray((filepresiones))
        tiempos=np.asarray((filetiempos))
        volumenes=np.asarray((filevolumenes)) 
        volumenes=volumenes
        
        
        FV=np.zeros((flujos.shape[0],2))
        FV[:,0]=flujos
        FV[:,1]=volumenes
        reg = linear_model.LinearRegression()
        reg.fit(FV,presiones)
        R,E=reg.coef_
        print(name)
        print('Resistence(R)=',(R,2), 'cm H2O L/S, ')
        #print('Compliance (C)=',round(1/E,4), 'L/cm H2O')
        print('Compliance (C)=',(1000*1/E), 'ml/cm H2O')

        print('---------------')
        P=E*volumenes+R*flujos
    #    plt.plot(tiempos,P)
    #    plt.plot(tiempos,presiones)
        erre = (R)
        ce = (1000*1/E)
        Y_data[i,0] = erre
        Y_data[i,1] = ce

        return


    #leemos npy generado de ambos pulmones
    maflujos=np.load(lado+model+str(ii)+'fluxes.npy')*60
    mapresiones=np.load(lado+model+str(ii)+'presionestodas.npy')
    matiempos=np.load(lado+model+str(ii)+'tiempos.npy')
    mavolumenes=np.load(lado+model+str(ii)+'volumenes.npy')

    #Agregamos punto 0
    maflujos=np.concatenate((np.array([0]),np.asarray(maflujos)))
    mapresiones=np.concatenate((np.array([0]),np.asarray(mapresiones)))
    matiempos=np.concatenate((np.array([0]),np.asarray(matiempos)))
    mavolumenes=np.concatenate((np.array([0]),np.asarray(mavolumenes)))


    regression(maflujos,mapresiones,matiempos,mavolumenes,name)

    #Y_train[i,0] = erre
    #Y_train[i,1] = ce

    #print(X_train[i], Y_train[i])

    i += 1

print(X_data, Y_data)

np.save('input_data.npy', X_data)
np.save('output_data.npy', Y_data)

print("--- %s seconds ---" % (time.time() - start_time))