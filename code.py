from modern_robotics import *
import numpy as np
import csv
import pandas as pd

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    joint_vector_list = []

    # Iteration 0 --------------------------------------------------------------------
    Vb = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                      thetalist)), T)))
    omega_b_norm = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    v_b_norm = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    err = omega_b_norm > eomg \
          or v_b_norm > ev

    joint_vector_list.append(np.array(thetalist))

    # Logging------------------------------------------------------------------------
    print(f"Iteration {i}:\n")
    print(f"joint vector:\n {thetalist}", sep="\n", end="\n\n")
    print(f"SE(3) end-effector config:\n {FKinBody(M, Blist, thetalist)}", sep="\n", end="\n\n")
    print(f"error twist V_b:\n {Vb}", sep="\n", end="\n\n")
    print(f"angular error magnitude ||omega_b||: {omega_b_norm}", sep="\n", end="\n\n")
    print(f"linear error magnitude ||v_b||: {v_b_norm}", sep="\n", end="\n\n")
    print("_"*70)


    while err and i < maxiterations:
        thetalist = np.round(thetalist \
                    + np.dot(np.linalg.pinv(JacobianBody(Blist, \
                                                         thetalist)), Vb), 4)
        i = i + 1
        Vb \
        = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                       thetalist)), T)))
        omega_b_norm = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        v_b_norm = np.linalg.norm([Vb[3], Vb[4], Vb[5]])

        err = omega_b_norm > eomg \
              or v_b_norm > ev

        #Logging
        print(f"Iteration {i}:\n")
        print(f"joint vector:\n {thetalist}", sep="\n", end="\n\n")
        print(f"SE(3) end-effector config:\n {FKinBody(M, Blist, thetalist)}", sep="\n", end="\n\n")
        print(f"error twist V_b:\n {Vb}", sep="\n", end="\n\n")
        print(f"angular error magnitude ||omega_b||: {omega_b_norm}", sep="\n", end="\n\n")
        print(f"linear error magnitude ||v_b||: {v_b_norm}", sep="\n", end="\n\n")
        print("_" * 70)

        joint_vector_list.append(np.array(thetalist))

    if not err:
        print(f"The algorithm has converged in {i} iterations")
    else:
        print(f"The algorithm has exceeded the maximum number of iterations (20)")

    df_indexed = pd.DataFrame(list(zip(*joint_vector_list))).add_prefix('theta')
    df_indexed.to_csv('iterates_indexed.csv', index_label='iteration')

    df = pd.DataFrame(list(zip(*joint_vector_list)))
    df.to_csv('iterates.csv', index=False)

    print("Csv file 'iterates.csv' has been created")

    return (thetalist, not err)
