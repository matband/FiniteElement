
def configFEM(nodes,elements,elementProperties,materials,constrained_dofs,point_loads,surf_forces,body_forces,thickness,filename):
    # Generate the text file
    with open(filename+'.txt', 'w') as f:
        # Quantities
        f.write("#FEM Config File\n")
        f.write("#Quantities\n")
        f.write('#Nodes/Elements/Materials/Degs of Contraint/Point Loads/Surface Forces/Body Forces\n')
        f.write(f"{len(nodes)} {len(elements)} {len(materials)} {len(constrained_dofs)} {len(point_loads)} {len(surf_forces)} {len(body_forces)}\n")

        # Node List
        f.write("#Node List - x,y\n")
        for node in nodes:
            f.write(f"{node[0]} {node[1]}\n")

        # Elem List
        f.write("#Elem List - material/type/n of nodes/list of nodes\n")
        for i,element in enumerate(elements):
            f.write(f"{elementProperties[i][0]} {elementProperties[i][1]} {elementProperties[i][2]} ")
            f.write(" ".join(map(str, element)))
            f.write("\n")

        # Material List
        f.write("#Material List - E/nu/rho\n")
        for material in materials:
            f.write(f"{material[0]} {material[1]} {material[2]}\n")

        # Constrained Dofs List
        f.write("#Constrained Dofs List - Node id / dof -1 if all dofs constrained on node\n")
        for dof in constrained_dofs:
            f.write(f"{dof[0]} {dof[1]}\n")

        # Applied Point Loads List
        f.write("#Applied Point Loads List - Node id / dof / value\n")
        for load in point_loads:
            f.write(f"{load[0]} {load[1]} {load[2]}\n")

        # Applied Surface Forces List
        f.write("#Applied Surface Forces List -  Element Id / Edge ID / Direction /value\n")
        for force in surf_forces:
            f.write(f"{force[0]} {force[1]} {force[2]} {force[3]}\n")

        # Applied Body Forces List
        f.write("#Applied Body Forces List - Direction / value\n")
        for force in body_forces:
            f.write(f"{force[0]} {force[1]}\n")

        # Thickness
        f.write("#Thickness (Plane Stress) - value - if zero: Plane Strain \n")
        f.write(f"{thickness}\n")

    f.close()