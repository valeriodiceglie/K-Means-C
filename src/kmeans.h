/**
* @author Valerio Di Ceglie
*/

#ifndef KMEANS_H
#define KMEANS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>

/**
 * @brief Enumeration for the initialization type
 * @enum RANDOM_DATAPOINT select k random data points as initial centroid
 * @enum KMEANSPP Choose the first centroid uniformly at random. Then, compute the distance squared
 * probabilities for each data point related to the first centroid. Loop for each remaining centroid
 * */
typedef enum initType_e {
    RANDOM_DATAPOINT = 0,
    KMEANSPP = 1
} init_method;

/**
 * @brief Enumeration for the type of distance to compute
 * @enum EUCLIDEAN
 * @enum MANHATTAN
 * */
typedef enum DistanceType_e {
    EUCLIDEAN = 0,
    MANHATTAN = 1
} distance_type;

/**
 * @brief Structure to represent a cluster
 * */
typedef struct Cluster_s {
    /** An array containing the medoid coordinates */
    float *centroid;

    /** An array that contains data point indexes belonging to corresponding cluster */
    unsigned int *data_points;

    /** Number of points that belong to cluster*/
    unsigned int num_points;
} Cluster;

/**
 * @brief Struct for the kmedoid initial configuration
 * */
typedef struct kmeans_config_s {

    /** The number of cluster */
    unsigned int k;

    /** The number of objects in the clustering space*/
    unsigned int num_objs;

    /** The number of dimension of the dataset entries*/
    unsigned int dim;

    /** Array of pointers to the clustering space */
    float **objs;

    /** Array of clusters */
    Cluster *clusters;

    /** Numbers of iteration to reach a stable solution*/
    unsigned int iters;

    /** Silhouette score for the current configuration*/
    float silhouette_score;

    /** Initialization method
     * @enum 0 --- RANDOM DATAPOINT
     * @enum 1 --- K-MEANS++
     * */
    unsigned int init_method;

    /**
     * @brief Distance type
     * @enum 0 --- EUCLIDEAN
     * @enum 1 --- MANHATTAN
     */
    unsigned int distance_type;

    /** Number of maximum iterations */
    unsigned int max_iters;
 } kmeans_config;

 // ------------------- PUBLIC DECL -------------------------

/**
 * @brief Print the cluster centroids and his data points
 * @param config the configuration instance
 */
void print_clusters(kmeans_config *config);

/**
 * @brief Compute the K-Means algorithm
 * @param config
 */
void kmeans(kmeans_config *config);

/**
 * @brief Get the cluster id which detain the datapoint
 * @param config the configuration instance
 * @param idx_datapoint the datapoint index
 * @return The cluster id which detain the datapoint
 */
unsigned int get_cluster_idx(kmeans_config *config, unsigned int idx_datapoint);

/**
 * @brief Deinitialize the K-Means configuration
 * @param config the configuration instance
 */
void kmeans_deinit(kmeans_config *config);

#endif // KMEANS_H
