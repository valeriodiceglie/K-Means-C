/**
* @author Valerio Di Ceglie
*/

#include "kmeans.h"

static void initialize_clusters(kmeans_config *config) {

    init_method init_cond = config->init_method;

    #ifdef _WIN32
        srand(time(NULL));
    #endif

    #ifdef __unix__
        unsigned int seed;
        FILE *f;
        f = fopen("/dev/random", "r");
        fread(&seed, sizeof(seed), 1, f);
        fclose(f);
        srand(seed);
    #endif

    if(init_cond == RANDOM_DATAPOINT) {
        for(unsigned int i = 0; i < config->k; i++) {
            config->clusters[i].centroid = malloc(sizeof(float) * config->dim);
            unsigned int random_idx = (unsigned int) rand() / (RAND_MAX / config->num_objs);
            memcpy(config->clusters[i].centroid, config->objs[random_idx],
                   sizeof(float) * config->dim);
            config->clusters[i].num_points = 0;
            config->clusters[i].data_points = (unsigned int *) malloc(sizeof(unsigned int) *
                                                                   config->num_objs);
        }
    }
    else if (init_cond == KMEANSPP) {
        unsigned int n_clusters  = config->k;
        unsigned int sample_size = config->num_objs;
        unsigned int dimension = config->dim;

        // Randomly select the first centroid
        unsigned int first_centroid_idx       = rand() % sample_size;
        config->clusters[0].centroid       = malloc(sizeof(float) * dimension);
        memcpy(config->clusters[0].centroid, config->objs[first_centroid_idx],
               sizeof(float) * dimension);
        config->clusters[0].num_points  = 0;
        config->clusters[0].data_points = malloc(sizeof(unsigned int) * sample_size);

        unsigned int n_local_trials = 2 + (int) log(n_clusters);

        /*
        * distance_to_closest     -> Array to store the squared distances to the closest centroid for
        *                            each point
        * distances_to_candidates -> Matrix for the squared distances to the centroid candidates
        * new_dist                -> Array to store the minimum between the closest centroid distance
        *                            and the corresponding distance to centroid candidates
        * rand_val                -> Array to store the generated random value
        * centroid_idxs           -> Array to store the centroid indexes, which are selected with a
        *                            probability distribution proportional to the squared distances
        *                            of the previous selected centroids
         */

        float *distances_to_closest = (float *) malloc(sizeof(float) * sample_size);
        float **distances_to_candidates = (float **) malloc(sizeof(float *) *
                                                                      sample_size);
        float *new_dist = (float *) malloc(sizeof(float) * sample_size);
        float new_pot;
        float *rand_val  = malloc(sizeof(float) * n_local_trials);
        unsigned int *centroid_idxs = malloc(sizeof(unsigned int) * n_local_trials);

        float sum_distances = 0.0f;

        for(unsigned int i = 0; i < sample_size; i++) {
            distances_to_candidates[i] = malloc(sizeof(float) * n_local_trials);
        }

        /*
         * Initialize distances to closest centroid (that is the distance between the datapoint and
         * the first centroid)
         */
        for(unsigned int i = 0; i < sample_size; i++) {
            distances_to_closest[i] = distance(config,
                                               config->clusters[0].centroid,
                                               config->objs[i]) *
                                      distance(config,
                                               config->clusters[0].centroid,
                                               config->objs[i]);
            sum_distances += distances_to_closest[i];
        }

        // Select the remaining centroids
        for(unsigned int i = 1; i < n_clusters; i++) {
            for(unsigned int k = 0; k < n_local_trials; k++) {
                rand_val[k] = ((float) rand() / (float) (RAND_MAX)) * sum_distances;
            }

            // Compute cumulative sum and search for candidate indexes
            // printf("current sum distances: %.3lf\n", sum_distances);
            for(unsigned int k = 0; k < n_local_trials; k++) {
                float cumulative_distances = 0.0f;
                // printf("rand val[%d]: %.3lf ", k, rand_val[k]);
                for(unsigned int j = 0; j < sample_size; j++) {
                    cumulative_distances += distances_to_closest[j];
                    if(cumulative_distances > rand_val[k]) {
                        centroid_idxs[k] = j;
                        // printf(" --- index datapoint as potential centroid: %d\n", j);
                        break;
                    }
                }
            }

            // Compute squared distances to centroid candidates
            for(unsigned int j = 0; j < sample_size; j++) {
                for(unsigned int k = 0; k < n_local_trials; k++) {
                    distances_to_candidates[j][k] = distance(config,
                                                             config->objs[centroid_idxs[k]],
                                                             config->objs[j]) *
                                                    distance(config,
                                                             config->objs[centroid_idxs[k]],
                                                             config->objs[j]);
                }
            }

            unsigned int best_candidate;
            float best_pot          = FLT_MAX;
            float *best_distance_sq = malloc(sizeof(float) * sample_size);

            for(unsigned int trial = 0; trial < n_local_trials; trial++) {
                new_pot = 0.0f;

                for(unsigned int j = 0; j < sample_size; j++) {
                    new_dist[j] = fminf(distances_to_closest[j], distances_to_candidates[j][trial]);
                    new_pot += new_dist[j];
                }

                if(new_pot < best_pot) {
                    best_pot       = new_pot;
                    best_candidate = centroid_idxs[trial];
                    memcpy(best_distance_sq, new_dist, sizeof(float) * sample_size);
                }
            }
            // printf("Selected index centroid: %d\n", best_candidate);
            sum_distances = best_pot;
            memcpy(distances_to_closest, best_distance_sq, sizeof(float) * sample_size);

            config->clusters[i].centroid       = malloc(sizeof(float) * dimension);
            memcpy(config->clusters[i].centroid, config->objs[best_candidate],
                   sizeof(float) * dimension);
            config->clusters[i].num_points  = 0;
            config->clusters[i].data_points = malloc(sizeof(unsigned int) * sample_size);
            free(best_distance_sq);
        }

        // Cleanup allocated memory
        for(unsigned int i = 0; i < sample_size; i++) {
            free(distances_to_candidates[i]);
        }
        free(rand_val);
        free(centroid_idxs);
        free(distances_to_closest);
        free(distances_to_candidates);
        free(new_dist);
    }
}

static float distance(kmeans_config *config, float *a, float *b){

    distance_type distance_t = config->distance_type;
    float dist = 0.0f;

    if(distance_t == EUCLIDEAN) {
        for(unsigned int i = 0; i < config->dim; i++) {
            float x1 = a[i];
            float x2 = b[i];
            dist += (x1 - x2) * (x1 - x2);
        }
        return sqrtf(dist);
    }
    else if(distance_t == MANHATTAN) {
        for(unsigned int i = 0; i < config->dim; i++) {
            float x1 = a[i];
            float x2 = b[i];
            dist += fabsf(x1 - x2);
        }
        return dist;
    }
}

static void assign_points_to_cluster(kmeans_config *config) {

    bool *point_assigned = malloc(sizeof(bool) * config->num_objs);

    for(unsigned int i = 0; i < config->num_objs; i++) {
        point_assigned[i] = false;
    }

    for(unsigned int i = 0; i < config->k; i++) {

        config->clusters[i].num_points = 0;

        free(config->clusters[i].data_points);

        config->clusters[i].data_points = (unsigned int *) malloc(sizeof(unsigned int) * config->num_objs);
    }

    for(unsigned int i = 0; i < config->num_objs; i++) {
        if(point_assigned[i]) { continue ; }

        float min_distance = FLT_MAX;
        unsigned int cluster_idx = 0;

        for(unsigned int j = 0; j < config->k; j++) {
            float d = distance(config, config->objs[i], config->clusters[j].centroid);
            if (d < min_distance) {
                cluster_idx = j;
                min_distance = d;
            }
        }

        config->clusters[cluster_idx].data_points[config->clusters[cluster_idx].num_points] = i;
        config->clusters[cluster_idx].num_points += 1;

        point_assigned[i] = true;
    }

    free(point_assigned);
}

static void update_centroids(kmeans_config *config) {
    for(unsigned int i = 0; i < config->k; i++) {
        for (unsigned int j = 0; j < config->dim; j++) {
            config->clusters[i].centroid[j] = 0.0f;
        }
        for(unsigned j = 0; j < config->clusters[i].num_points; j++) {
            for(unsigned int k = 0; k < config->dim; k++) {
                config->clusters[i].centroid[k] +=
                    config->objs[config->clusters[i].data_points[j]][k];
            }
        }
        for(unsigned int j = 0; j < config->dim; j++) {
            config->clusters[i].centroid[j] /= config->clusters[i].num_points;
        }
    }
}

static float silhouette_score(kmeans_config *config) {
    Cluster *curr_cluster;
    unsigned int curr_cluster_idx;
    float sum_a, sum_b, a, temp_b, s;
    float b = FLT_MAX;

    // Compute s foreach datapoint
    for(unsigned int i = 0; i < config->k; i++) {

        // Find the current datpoint's cluster
        curr_cluster_idx = get_cluster_idx(config, i);
        curr_cluster = &config->clusters[curr_cluster_idx];

        // Compute intra-cluster distance
        sum_a = 0;
        for(unsigned int j = 0; j < curr_cluster->num_points; j++) {
            a = 0;

            if(curr_cluster->data_points[j] == i) { continue; }

            sum_a += distance(config, config->objs[i], config->objs[curr_cluster->data_points[j]]);
        }
        a = sum_a / (float) (curr_cluster->num_points - 1);

        // Compute inter-cluster distance
        sum_b = 0.0f;
        b = FLT_MAX;
        for(unsigned int j = 0; j < config->k; j++) {
            if (j == curr_cluster_idx) { continue; }
            for(unsigned int k = 0; k < config->clusters[j].num_points; k++) {
                sum_b += distance(config, config->objs[i],
                                  config->objs[config->clusters[j].data_points[k]]);
            }
            temp_b = sum_b / (float) config->clusters[j].num_points;
            if(temp_b < b) { b = temp_b; }
        }
        s += (b - a) / fmaxf(b, a);
    }
    return s / (float) config->num_objs;
}

void print_clusters(kmeans_config *config) {
    for(unsigned int i = 0; i < config->k; i++) {
        printf("Cluster %d: Centroid[", i);
        for (unsigned int j = 0; j < config->dim; j++) {
            printf("%.3lf  ", config->clusters[i].centroid[j]);
        }
        printf("]\n");
        printf("Number of points: %d\n", config->clusters[i].num_points);
    }
}

unsigned int get_cluster_idx(kmeans_config *config, unsigned int datapoint_idx) {
    for(unsigned int i = 0; i < config->k; i++) {
        for(unsigned int j = 0; j < config->clusters[i].num_points ; j++) {
            if(config->clusters[i].data_points[j] == datapoint_idx) { return i; }
        }
    }
}

void kmeans(kmeans_config *config) {
    unsigned int iterations = 0;
    unsigned int dim = config->dim;
    //float *mean = malloc(sizeof(float) * dim);


    /*for(ulib_uint i = 0; i < dim; i++) {
        for(ulib_uint j = 0; j < config->num_objs; j++) {
            mean[i] += uvec_get(ulib_float, config->objs[j], i);
        }
        mean[i] /= config->num_objs;
    }

    for(ulib_uint i = 0; i < dim; i++) {
        //printf("MEAN: [%d]: %.2lf\n", i, mean[i]);
        for(ulib_uint j = 0; j < config->num_objs; j++) {
            ulib_float curr_val = uvec_get(ulib_float, config->objs[j], i);
            uvec_set(ulib_float, config->objs[j], i, curr_val - mean[i]);
        }
    }*/

    initialize_clusters(config);
    printf("Initial Clusters: \n");
    print_clusters(config);

    while(iterations < config->max_iters) {

        assign_points_to_cluster(config);

        update_centroids(config);

        iterations++;
    }

    config->iters = iterations;
    config->silhouette_score = silhouette_score(config);
}

void kmeans_deinit(kmeans_config *config) {
    // Free array of index of datapoints
    for(unsigned int i = 0; i < config->k; i++) {
        free(config->clusters[i].data_points);
        free(config->clusters[i].centroid);
    }
}
