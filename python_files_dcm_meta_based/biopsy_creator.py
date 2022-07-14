import numpy as np
import matplotlib.pyplot as plt

def biopsy_points_creater_ring(list_centroid_line_vector,list_origin_to_first_centroid_vector):
    """
    A function that creates points in a cylinder surrounding the centroid line, 
    note that the centroid vector that describes the line must point in the direction
    of the centroid line
    """
    centroid_vector = np.array(list_centroid_line_vector)
    norm_centroid_vector = np.linalg.norm(centroid_vector)
    lab_polar_centroid = np.arccos(centroid_vector[2]/norm_centroid_vector)
    norm_centroid_vector_xy_proj = norm_centroid_vector*np.sin(lab_polar_centroid)
    lab_azimuth_centroid = np.arccos(centroid_vector[0]/norm_centroid_vector_xy_proj)
    if centroid_vector[1] == 0 and centroid_vector[0] == 0:
        lab_azimuth_centroid = 0
    if centroid_vector[1] < 0:
        lab_azimuth_centroid = 2*np.pi - lab_azimuth_centroid
    centroid_x = np.sin(lab_polar_centroid)*np.cos(lab_azimuth_centroid)
    centroid_y = np.sin(lab_polar_centroid)*np.sin(lab_azimuth_centroid)
    centroid_z = np.cos(lab_polar_centroid)


    # plot the stuff? if not confident that algo works then change below plot var to True
    plot_stuff = True
    if plot_stuff == True:
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        unit_centroid_vector = (1/norm_centroid_vector)*centroid_vector
        ax.scatter(unit_centroid_vector[0], unit_centroid_vector[1], unit_centroid_vector[2], c='r', marker='o')
        ax.scatter(centroid_x, centroid_y, centroid_z, c='b', marker='x')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        ax.set_zlim(-2,2)

        
        #print(list_centroid_line_vector)
        #print(unit_centroid_vector)
        #print('azimuth centroid = ',lab_azimuth_centroid)
        #print('polar centroid = ',lab_polar_centroid)
        #print(centroid_x,centroid_y,centroid_z)
        plt.show()



    origin_to_first_centroid_vector = np.array(list_origin_to_first_centroid_vector)

    theta_0 = lab_polar_centroid # polar angle of the incoming photon in the lab frame
    phi_0 = lab_azimuth_centroid # azimuth angle of the incoming photon in the lab frame
    chi = np.pi/2 # in-plane scatter angle of the outgoing photon relative to the incoming photon vector
    
    num_ring_points = 20
    lab_ring_points = np.empty(shape=[num_ring_points,3])

    
    rotation_matrix_x = np.array([np.cos(theta_0)*np.cos(phi_0),-np.sin(phi_0),np.sin(theta_0)*np.cos(phi_0)])
    rotation_matrix_y = np.array([np.cos(theta_0)*np.sin(phi_0),np.cos(phi_0),np.sin(theta_0)*np.sin(phi_0)])
    rotation_matrix_z = np.array([-np.sin(theta_0),0,np.cos(theta_0)])
    rotation_matrix = rotation_matrix_x
    rotation_matrix = np.vstack([rotation_matrix,rotation_matrix_y])
    rotation_matrix = np.vstack([rotation_matrix,rotation_matrix_z])

    for j in range(0,num_ring_points):
        eta = 0+j*2*np.pi/num_ring_points # out of plane scatter angle of the outgoing photon relative to the incoming photon vector

        centroid_vec_frame_vec_to_circ_x = np.array([np.sin(chi)*np.cos(eta)])
        centroid_vec_frame_vec_to_circ_y = np.array([np.sin(chi)*np.sin(eta)])
        centroid_vec_frame_vec_to_circ_z = np.array([np.cos(chi)])
        centroid_vec_frame_vec_to_circ = centroid_vec_frame_vec_to_circ_x
        centroid_vec_frame_vec_to_circ = np.vstack([centroid_vec_frame_vec_to_circ,centroid_vec_frame_vec_to_circ_y])
        centroid_vec_frame_vec_to_circ = np.vstack([centroid_vec_frame_vec_to_circ,centroid_vec_frame_vec_to_circ_z])

        lab_vec_to_circle = np.dot(rotation_matrix,centroid_vec_frame_vec_to_circ)

        # create point, transport to first centroid, then go to orthogonal ring
        lab_ring_point = origin_to_first_centroid_vector + lab_vec_to_circle.T
        #print(lab_ring_point)
        lab_ring_points[j] = lab_ring_point

    
    lab_ring_points_Transpose = lab_ring_points.T
    
    # transport and create 
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unit_centroid_vector = (1/norm_centroid_vector)*centroid_vector
    ax.scatter(origin_to_first_centroid_vector[0], origin_to_first_centroid_vector[1], origin_to_first_centroid_vector[2], c='g', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    ax.set_zlim(-2,2)
    ax.scatter(lab_ring_points_Transpose[0], lab_ring_points_Transpose[1], lab_ring_points_Transpose[2], c='r', marker='o')
    plt.show()
    #print('done!')

def biopsy_points_creater_by_transport(list_centroid_line_vector,list_origin_to_first_centroid_vector,num_centroids,centroid_separation_distance,plot_stuff):
    """
    A function that creates points in a cylinder surrounding the centroid line, 
    note that the centroid vector that describes the line must point from the 
    first centroid point towards the rest of them, since at the moment this 
    programme works by creating one ring, and then transporting this ring 
    forwards to make successive rings
    """
    centroid_vector = np.array(list_centroid_line_vector)
    norm_centroid_vector = np.linalg.norm(centroid_vector)
    lab_polar_centroid = np.arccos(centroid_vector[2]/norm_centroid_vector)
    norm_centroid_vector_xy_proj = norm_centroid_vector*np.sin(lab_polar_centroid)
    lab_azimuth_centroid = np.arccos(centroid_vector[0]/norm_centroid_vector_xy_proj)
    if centroid_vector[1] == 0 and centroid_vector[0] == 0:
        lab_azimuth_centroid = 0
    if centroid_vector[1] < 0:
        lab_azimuth_centroid = 2*np.pi - lab_azimuth_centroid
    centroid_x = np.sin(lab_polar_centroid)*np.cos(lab_azimuth_centroid)
    centroid_y = np.sin(lab_polar_centroid)*np.sin(lab_azimuth_centroid)
    centroid_z = np.cos(lab_polar_centroid)
    

    # plot the stuff? if not confident that algo works then change below plot var to True
    unit_centroid_vector = (1/norm_centroid_vector)*centroid_vector

    
    if plot_stuff == True:
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(unit_centroid_vector[0], unit_centroid_vector[1], unit_centroid_vector[2], c='r', marker='o')
        ax.scatter(centroid_x, centroid_y, centroid_z, c='b', marker='x')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        ax.set_zlim(-2,2)

        
        #print(list_centroid_line_vector)
        #print(unit_centroid_vector)
        #print('azimuth centroid = ',lab_azimuth_centroid)
        #print('polar centroid = ',lab_polar_centroid)
        #print(centroid_x,centroid_y,centroid_z)
        plt.show()



    origin_to_first_centroid_vector = np.array(list_origin_to_first_centroid_vector)

    theta_0 = lab_polar_centroid # polar angle of the incoming photon in the lab frame
    phi_0 = lab_azimuth_centroid # azimuth angle of the incoming photon in the lab frame
    chi = np.pi/2 # in-plane scatter angle of the outgoing photon relative to the incoming photon vector
    
    num_ring_points = 20
    lab_ring_points = np.empty(shape=[num_ring_points*num_centroids,3])

    
    rotation_matrix_x = np.array([np.cos(theta_0)*np.cos(phi_0),-np.sin(phi_0),np.sin(theta_0)*np.cos(phi_0)])
    rotation_matrix_y = np.array([np.cos(theta_0)*np.sin(phi_0),np.cos(phi_0),np.sin(theta_0)*np.sin(phi_0)])
    rotation_matrix_z = np.array([-np.sin(theta_0),0,np.cos(theta_0)])
    rotation_matrix = rotation_matrix_x
    rotation_matrix = np.vstack([rotation_matrix,rotation_matrix_y])
    rotation_matrix = np.vstack([rotation_matrix,rotation_matrix_z])

    # transport and create more rings, first make an appropriately sized array for all points that are values of the first ring
    for k in range(0,num_centroids):
        for j in range(0,num_ring_points):
            eta = 0+j*2*np.pi/num_ring_points # out of plane scatter angle of the outgoing photon relative to the incoming photon vector

            centroid_vec_frame_vec_to_circ_x = np.array([np.sin(chi)*np.cos(eta)])
            centroid_vec_frame_vec_to_circ_y = np.array([np.sin(chi)*np.sin(eta)])
            centroid_vec_frame_vec_to_circ_z = np.array([np.cos(chi)])
            centroid_vec_frame_vec_to_circ = centroid_vec_frame_vec_to_circ_x
            centroid_vec_frame_vec_to_circ = np.vstack([centroid_vec_frame_vec_to_circ,centroid_vec_frame_vec_to_circ_y])
            centroid_vec_frame_vec_to_circ = np.vstack([centroid_vec_frame_vec_to_circ,centroid_vec_frame_vec_to_circ_z])

            lab_vec_to_circle = np.dot(rotation_matrix,centroid_vec_frame_vec_to_circ)

            # create point, transport to first centroid, then go to orthogonal ring
            lab_ring_point = origin_to_first_centroid_vector + lab_vec_to_circle.T
            #print(lab_ring_point)
            lab_ring_points[j+k*num_ring_points] = lab_ring_point

    # now add appropriate multiples of transport vector to each array slice, to make shifted rings
    for k in range(1,num_centroids):
        lab_ring_points[k*num_ring_points:(k+1)*num_ring_points] = lab_ring_points[(k-1)*num_ring_points:k*num_ring_points] + unit_centroid_vector*centroid_separation_distance

    lab_ring_points_Transpose = lab_ring_points.T
    
    
    
    
        
    if plot_stuff == True:
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        
        ax.scatter(origin_to_first_centroid_vector[0], origin_to_first_centroid_vector[1], origin_to_first_centroid_vector[2], c='g', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        ax.set_zlim(-2,2)
        ax.scatter(lab_ring_points_Transpose[0], lab_ring_points_Transpose[1], lab_ring_points_Transpose[2], c='r', marker='o')
        plt.show()
    #print('done!')

    return lab_ring_points_Transpose
    """
    # determine vector from centroid to draw circle using rotation matrix to find vector in lab frame from chi and eta realative to centroid vector 
    term1_x = np.cos(theta_0)*np.cos(phi_0)*np.sin(chi)*np.cos(eta)
    term2_x = -np.sin(phi_0)*np.sin(chi)*np.sin(eta)
    term3_x = np.sin(theta_0)*np.cos(phi_0)*np.cos(chi)
    vec_to_circle_x = term1_x+term2_x+term3_x
    
    term1_y = np.cos(theta_0)*np.sin(phi_0)*np.sin(chi)*np.cos(eta)
    term2_y = np.cos(phi_0)*np.sin(chi)*np.sin(eta)
    term3_y = np.sin(theta_0)*np.sin(phi_0)*np.cos(chi)
    vec_to_circle_y = term1_y+term2_y+term3_y

    term1_z = -np.sin(theta_0)*np.sin(chi)*np.cos(eta)
    term2_z = 0
    term3_z = np.cos(theta_0)*np.cos(chi)
    vec_to_circle_z = term1_z+term2_z+term3_z

    vec_to_circle = np.array([vec_to_circle_x,vec_to_circle_y,vec_to_circle_z])

    
    


    
    # determine polar angle in lab frame  
    polar_lab_photon = np.arccos(-np.sin(theta_0)*np.sin(chi)*np.cos(eta)+np.cos(theta_0)*np.cos(chi))
    
    # determine azimuth angle in lab frame
    term1 = np.cos(theta_0)*np.cos(phi_0)*np.sin(chi)*np.cos(eta)
    term2 = np.sin(phi_0)*np.sin(chi)*np.sin(eta)
    term3 = np.sin(theta_0)*np.cos(phi_0)*np.cos(chi)
    factor = np.sin(polar_lab_photon)
    #shift2 = round(rand*pi);
    
    azimuth_lab_photon = np.arccos((term1-term2+term3)/factor)


    # pick negative azimuth 50% of the time and positive 50% of the
    # time to deal with limited range of arccos
    #picker = rand;
    #if picker < 0.5 
    #    azimuth_lab_photon = acos((term1-term2+term3)/factor);#+shift2;
    #else
    #    azimuth_lab_photon = -acos((term1-term2+term3)/factor);#+shift2;
    #end 

    """


def tester():
    #centroid_list = [[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[-1,-1,-1],[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]
    #for centroid in centroid_list:
    #    biopsy_points_creater(centroid)
    centroid_1 = [1,1,1]
    centroid_2 = [3,4,7]
    centroid_vec = [centroid_2[x]-centroid_1[x] for x in range(0,3)]
    biopsy_points_creater_by_transport(centroid_vec,centroid_1,15,1,True)

def main():
    tester()


if __name__ == '__main__':    
    main()