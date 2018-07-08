# coding: utf-8

import numpy as np
import collections
import pandas as pd
import timeit
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter
from sklearn.cluster import AffinityPropagation
import multiprocessing as mp


def createUMIsWithBias(n, umi_size, efficiency_min, efficiency_max):
    """
    this function creates a list of random UMIs

    assumption is that UMIs are truly random, e.g equal probability of
    each base at each position. This is not actually the case(!)

    Also now returns a dictionary of simulated efficiencies

    :param n: number of UMIs used for simulation
    :param umi_size: number of bases in UMI sequence
    :param efficiency_min: minimum efficiency of PCR amplification for one UMI sequence
    :param efficiency_max: maximum efficiency of PCR amplification for one UMI sequence
    :return: Counter object representing pool of UMIs used for simulation
    """
    assert efficiency_min > 0 and efficiency_max <= 1, "PCR efficiencies must be between 0 and 1"

    UMIs = []
    efficiency_dict = collections.defaultdict(float)
    for umi in range(0, n):
        UMI = ""
        for base in range(0, umi_size):
            UMI += np.random.choice(["A", "C", "G", "T"])

        UMIs.append(UMI)
        efficiency_dict[UMI] = np.random.uniform(efficiency_min, efficiency_max)

    return Counter(UMIs), efficiency_dict


def createTemplateUMIsWithBias(n, umi_size, efficiency_min, efficiency_max):
    """
    this function creates a list of random UMIs

    assumption is that UMIs are truly random, e.g equal probability of
    each base at each position. This is not actually the case(!)

    Also now returns a dictionary of simulated efficiencies

    :param n: number of UMIs used for simulation
    :param umi_size: number of bases in UMI sequence
    :param efficiency_min: minimum efficiency of PCR amplification for one UMI sequence
    :param efficiency_max: maximum efficiency of PCR amplification for one UMI sequence
    :return: Counter object representing pool of UMIs used for simulation
    """
    assert efficiency_min > 0 and efficiency_max <= 1, "PCR efficiencies must be between 0 and 1"
    assert umi_size == 9

    UMIs = []
    efficiency_dict = collections.defaultdict(float)
    for umi in range(0, n):
        UMI = ""
        UMI += np.random.choice(["A", "T"])
        UMI += np.random.choice(["G", "C"])
        UMI += np.random.choice(["A", "C", "T", "G"])
        UMI += np.random.choice(["A", "C", "T", "G"])
        UMI += np.random.choice(["A", "T"])
        UMI += np.random.choice(["G", "C"])
        UMI += np.random.choice(["A", "C", "T", "G"])
        UMI += np.random.choice(["A", "C", "T", "G"])
        UMI += np.random.choice(["A", "C", "G"])

        UMIs.append(UMI)
        efficiency_dict[UMI] = np.random.uniform(efficiency_min, efficiency_max)

    return Counter(UMIs), efficiency_dict


def transcribeUMIsWithErrors(umi_counter, error_rate, UMI_length):
    """
    Introduce errors into the UMI pool using poisson sampling based on an enzyme error rate.

    This function is used both for in vitro transcription and reverse transcription simulations.

    :param umi_counter: Counter object representing pool of UMIs input to the in vitro transcription reaction
    :param error_rate: per base error rate of enzyme (RNA polymerase or transcriptase)
    :param UMI_length: number of bases in UMI sequence
    :return: Counter object representing pool of UMIs after forward/reverse transcription
    """
    errors_dict = {"A": ["C", "G", "T"],
                   "C": ["A", "G", "T"],
                   "G": ["C", "A", "T"],
                   "T": ["C", "G", "A"]}
    UMI_pool_size = sum(umi_counter.values())
    UMI_pool = np.array(list(umi_counter.elements()))

    sample_num = np.random.poisson(error_rate * UMI_pool_size * UMI_length, 1)
    mutated_UMI_indices = np.random.choice(range(UMI_pool_size), sample_num, replace=False)
    mutated_UMIs_list = np.array([])

    if error_rate == 0:
        return umi_counter
    else:
        for umi_index in mutated_UMI_indices:
            # Replace base instead of appending (randomly select base position and replace by random base (AGTB)
            random_base = np.random.choice(range(UMI_length))
            umi = list(UMI_pool[umi_index])
            umi[random_base] = np.random.choice(errors_dict[umi[random_base]])
            umi = "".join(umi)
            mutated_UMIs_list = np.append(mutated_UMIs_list, umi)
        UMI_pool[mutated_UMI_indices] = mutated_UMIs_list
        return Counter(UMI_pool)


def TranscriptionEventsWithErrors(umi_counter, error_rate, transcription_events, umi_length):
    """
    Simulate in vitro transcription with errors by first amplifying the UMI pool
    based on the number of transcription events, then introducing errors into UMIs
    in the amplified pool using poisson sampling based on the error_rate

    :param umi_counter: Counter object representing pool of UMIs input to the in vitro transcription reaction
    :param error_rate: per base error rate of T7 RNA polymerase
    :param transcription_events: average number of UMI copies generated (transcription events for each UMI
    are sampled from a normal distribution with mean = transcription_events and s = 0.1 * mean)
    :param umi_length: number of bases in the UMI sequence
    :return: Counter object representing pool of UMIs after in vitro transcription
    """
    std = int(round(0.1 * transcription_events))
    for umi in umi_counter.keys():
        umi_counter[umi] = umi_counter[umi] * int(np.random.normal(transcription_events, std, 1))

    umi_counter = transcribeUMIsWithErrors(umi_counter, error_rate, umi_length)
    return umi_counter


def downsampleUMIs(umi_counter, efficiency=0.6):
    """
    subsample from UMI pool using an efficiency factor

    :param umi_counter: Counter object representing amplified pool of UMIs to be downsampled (e.g. after
    in vitro transcription of PCR amplification)
    :param efficiency: proportion of UMIs left after downsampling
    :return: Counter object representing downsampled pool of UMIs
    """
    diluted_size = int(round(sum(umi_counter.values()) * efficiency))
    p_vals_UMI = np.divide(list(umi_counter.values()), sum(umi_counter.values()))
    sampled_UMIs = np.random.choice(list(umi_counter.keys()), size=diluted_size, p=p_vals_UMI)
    sampled_UMIs = Counter(sampled_UMIs)
    return sampled_UMIs


# Cluster methods
def edit_dist(first, second):
    """
    returns the edit distance/hamming distances between two strings

    :param first: string 1
    :param second: string 2
    :return: hamming distance
    """

    dist = sum([not a == b for a, b in zip(first, second)])
    return dist


def dedup_naive(molecular_barcodes, mismatches=1):
    clusters_dict = {}
    nclusters = 0
    for i, molecular_barcode in enumerate(sorted(molecular_barcodes.keys())):
        if i == 0:
            clusters_dict[nclusters] = [molecular_barcode]
        else:
            # compare distant of previous molecular barcodes and new one
            # if distance is between threshold we add it to the cluster 
            # otherwise we create a new cluster
            if edit_dist(clusters_dict[nclusters][-1], molecular_barcode) <= mismatches:
                clusters_dict[nclusters].append(molecular_barcode)
            else:
                nclusters += 1
                clusters_dict[nclusters] = [molecular_barcode]
    return len(clusters_dict)


def hierarchical_single(molecular_barcodes, mismatches=1):
    """Deduplicate UMIs using single distance and fcluster methods
    """
    molecular_barcodes = list(molecular_barcodes.keys())

    def d(coord):
        i, j = coord
        return edit_dist(molecular_barcodes[i], molecular_barcodes[j])

    # Create hierarchical clustering and obtain flat clusters at the distance given
    indices = np.triu_indices(len(molecular_barcodes), 1)
    distance_matrix = np.apply_along_axis(d, 0, indices)
    linkage_cluster = linkage(distance_matrix, method="single")
    flat_clusters = fcluster(linkage_cluster, mismatches, criterion='distance')
    return len(set(flat_clusters))


def hierarchical_complete(molecular_barcodes, mismatches=1):
    """Deduplicate UMIs using complete distance and fcluster methods
    """
    molecular_barcodes = list(molecular_barcodes.keys())

    def d(coord):
        i, j = coord
        return edit_dist(molecular_barcodes[i], molecular_barcodes[j])

    # Create hierarchical clustering and obtain flat clusters at the distance given
    indices = np.triu_indices(len(molecular_barcodes), 1)
    distance_matrix = np.apply_along_axis(d, 0, indices)
    linkage_cluster = linkage(distance_matrix, method="complete")
    flat_clusters = fcluster(linkage_cluster, mismatches, criterion='distance')
    return len(set(flat_clusters))


def hierarchical_ward(molecular_barcodes, mismatches=1):
    """Deduplicate UMIs using ward's criteria and fcluster methods
    """
    molecular_barcodes = list(molecular_barcodes.keys())

    def d(coord):
        i, j = coord
        return edit_dist(molecular_barcodes[i], molecular_barcodes[j])

    # Create hierarchical clustering and obtain flat clusters at the distance given
    indices = np.triu_indices(len(molecular_barcodes), 1)
    distance_matrix = np.apply_along_axis(d, 0, indices)
    linkage_cluster = linkage(distance_matrix, method="ward")
    flat_clusters = fcluster(linkage_cluster, mismatches, criterion='distance')
    return len(set(flat_clusters))


def affinity(molecular_barcodes):
    """Deduplicate UMIs using the affinity propagation method
    """
    molecular_barcodes = list(molecular_barcodes.keys())
    words = np.asarray(molecular_barcodes)
    lev_similarity = -1 * np.array([[edit_dist(w1, w2) for w1 in words] for w2 in words])
    affprop = AffinityPropagation(affinity="precomputed", damping=0.8)
    affprop.fit(lev_similarity)
    unique_clusters = list()
    for cluster_id in np.unique(affprop.labels_):
        cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
        unique_clusters.append(cluster)
    return len(unique_clusters)


def dedup_none(Counter):
    """
    Count all UMIs as unique
    """
    return sum(Counter.values())


def dedup_unique(Counter):
    """
    Count all unique UMIs
    """
    return len(Counter.keys())


def dedup_percentile(Counter):
    """
    Remove UMIs with counts lower than 1% of the mean
    """
    threshold = np.mean(list(Counter.values())) / 100
    return len([umi for umi in list(Counter.keys()) if Counter[umi] > threshold])


def breadth_first_search(node, adj_list):
    searched = set()
    found = set()
    queue = set()
    queue.update((node,))
    found.update((node,))

    while len(queue) > 0:
        node = (list(queue))[0]
        found.update(adj_list[node])
        queue.update(adj_list[node])
        searched.update((node,))
        queue.difference_update(searched)

    return found


def dedup_cluster(Counter, mismatches=1):
    def get_adj_list_cluster(umis):
        return {umi: [umi2 for umi2 in umis if edit_dist(umi, umi2) <= mismatches] for umi in umis}

    def get_connected_components_cluster(graph, Counter):
        found = list()
        components = list()
        for node in sorted(graph, key=lambda x: Counter[x], reverse=True):
            if node not in found:
                component = breadth_first_search(node, graph)
                found.extend(component)
                components.append(component)
        return components

    adj_list = get_adj_list_cluster(list(Counter.keys()))
    clusters = get_connected_components_cluster(adj_list, Counter)
    return len(clusters)


def dedup_adj(Counter, mismatches=1):
    def get_adj_list_adjacency(umis):
        return {umi: [umi2 for umi2 in umis if edit_dist(umi, umi2) <= mismatches] for umi in umis}

    def get_connected_components_adjacency(graph, Counter):
        found = list()
        components = list()
        for node in sorted(graph, key=lambda x: Counter[x], reverse=True):
            if node not in found:
                component = breadth_first_search(node, graph)
                found.extend(component)
                components.append(component)
        return components

    def remove_umis(adj_list, cluster, nodes):
        '''removes the specified nodes from the cluster and returns
        the remaining nodes '''
        # list incomprehension: for x in nodes: for node in adj_list[x]: yield node
        nodes_to_remove = set([node
                               for x in nodes
                               for node in adj_list[x]] + nodes)
        return cluster - nodes_to_remove

    def get_best_adjacency(cluster, adj_list, counts):
        if len(cluster) == 1:
            return list(cluster)
        sorted_nodes = sorted(cluster, key=lambda x: counts[x],
                              reverse=True)
        for i in range(len(sorted_nodes) - 1):
            if len(remove_umis(adj_list, cluster, sorted_nodes[:i + 1])) == 0:
                return sorted_nodes[:i + 1]

    def reduce_clusters_adjacency(adj_list, clusters, counts):
        # TS - the "adjacency" variant of this function requires an adjacency
        # list to identify the best umi, whereas the other variants don't
        # As temporary solution, pass adj_list to all variants
        n = 0
        for cluster in clusters:
            parent_umis = get_best_adjacency(cluster, adj_list, counts)
            n += len(parent_umis)
        return n

    adj_list = get_adj_list_adjacency(list(Counter.keys()))
    clusters = get_connected_components_adjacency(adj_list, Counter)
    count = reduce_clusters_adjacency(adj_list, clusters, Counter)
    return count


def dedup_dir_adj(Counter, mismatches=1):
    def get_adj_list_directional_adjacency(umis, counts):
        return {umi: [umi2 for umi2 in umis if edit_dist(umi, umi2) <= mismatches and
                      counts[umi] >= (counts[umi2] * 2) - 1] for umi in umis}

    def get_connected_components_adjacency(graph, Counter):
        found = list()
        components = list()
        for node in sorted(graph, key=lambda x: Counter[x], reverse=True):
            if node not in found:
                component = breadth_first_search(node, graph)
                found.extend(component)
                components.append(component)
        return components

    def remove_umis(adj_list, cluster, nodes):
        '''removes the specified nodes from the cluster and returns
        the remaining nodes '''
        # list incomprehension: for x in nodes: for node in adj_list[x]: yield node
        nodes_to_remove = set([node
                               for x in nodes
                               for node in adj_list[x]] + nodes)
        return cluster - nodes_to_remove

    def reduce_clusters_directional_adjacency(adj_list, clusters, counts):
        n = 0
        for cluster in clusters:
            n += 1
        return n

    adj_list = get_adj_list_directional_adjacency(list(Counter.keys()), Counter)
    clusters = get_connected_components_adjacency(adj_list, Counter)
    count = reduce_clusters_directional_adjacency(adj_list, clusters, Counter)
    return count


def SimulateCycleWithErrors(UMI_counter, efficiency_dict, error_rate=0.00001, UMI_length=7):
    """
    simulate single PCR cycle for UMI pool with a UMI bias defined by the efficiency_dict

    :param UMI_counter: Counter object representing pool of UMIs input to a PCR cycle
    :param efficiency_dict: dictionary of variable efficiencies used to simulate amplification bias of UMIs
    :param error_rate: per base error rate of DNA polymerase used in the PCR reaction
    :param UMI_length: number of bases in the UMI sequence
    :return: Counter object representing pool of UMIs after single PCR cycle
    """
    errors_dict = {"A": ["C", "G", "T"],
                   "C": ["A", "G", "T"],
                   "G": ["C", "A", "T"],
                   "T": ["C", "G", "A"]}

    # Sample from poisson distribution with mu = efficiency*len(UMI)
    # Split efficiencies per umi
    # Iterate unique UMIs instead of all
    amp_list = []
    for umi in UMI_counter.keys():
        sample_umis = np.random.binomial(n=UMI_counter[umi], p=efficiency_dict[umi], size=1)
        amp_list += [umi] * int(sample_umis)
    amplified_UMIs = Counter(amp_list)

    if error_rate == 0:
        return UMI_counter
    else:
        # Sample from poisson distribution with mu = error_rate*len(UMI)*UMI_length
        sample_mutated = np.random.poisson(error_rate * sum(amplified_UMIs.values()) * UMI_length, 1)

        # mutated_UMIs = np.random.choice(UMI, sample_mutated, replace = False)
        p_vals_UMI = np.divide(list(amplified_UMIs.values()), sum(amplified_UMIs.values()))
        mutated_UMIs = np.random.choice(list(amplified_UMIs.keys()), size=sample_mutated, p=p_vals_UMI)

        mutated_UMIs_list = []
        for umi in mutated_UMIs:
            random_base = np.random.choice(range(len(umi)))
            umi = list(umi)
            umi[random_base] = np.random.choice(errors_dict[umi[random_base]])
            umi = "".join(umi)
            mutated_UMIs_list.append(umi)

        # Add mutated UMIs (no need to replace if len(pool) >> len(mutated))
        UMI_counter += amplified_UMIs
        UMI_counter += Counter(mutated_UMIs_list)
        return UMI_counter


def simulatePCRcycleWithErrorsAndBias(UMIs_counter, efficiency_min, efficiency_max,
                                      efficiency_dict, error_rate=0.00001, UMI_length=7):
    """
    update efficiency dictionary for new UMIs and run one PCR cycle by calling SimulateCycleWithErrors

    :param UMIs_counter: Counter object representing pool of UMIs input to the PCR reaction
    :param efficiency_min: minimum efficiency of PCR amplification for one UMI sequence
    :param efficiency_max: maximum efficiency of PCR amplification for one UMI sequence
    :param efficiency_dict: dictionary of variable efficiences used to simulate amplification bias of UMIs
    :param error_rate: per base error rate of DNA polymerase used in the PCR reaction
    :param UMI_length: number of bases in the UMI sequence
    :return:
    """
    # Define random probabilities of efficiency
    for umi in UMIs_counter.keys():
        if not efficiency_dict[umi]:
            efficiency_dict[umi] = np.random.uniform(efficiency_min, efficiency_max)
    new_list = SimulateCycleWithErrors(UMIs_counter, efficiency_dict, error_rate, UMI_length)
    return new_list


def PCRcyclesWithErrorsAndBias(umis_in, efficiency_min, efficiency_max,
                               efficiency_dict, PCR_cycles, error_rate=0.00001, UMI_length=7):
    """
    simulate PCR amplification of UMIs with replication errors and UMI bias

    :param umis_in: Counter object representing pool of UMIs input to the PCR reaction
    :param efficiency_min: minimum efficiency of PCR amplification for one UMI sequence
    :param efficiency_max: maximum efficiency of PCR amplification for one UMI sequence
    :param efficiency_dict: dictionary of variable efficiences used to simulate amplification bias of UMIs
    :param PCR_cycles: number of PCR cycles used for amplification
    :param error_rate: per base error rate of DNA polymerase used in the PCR reaction
    :param UMI_length: number of bases in the UMI sequence
    :return: Counter object representing pool of UMIs after PCR amplification
    """
    for cycle in range(0, PCR_cycles):
        post_cycle = simulatePCRcycleWithErrorsAndBias(umis_in, efficiency_min, efficiency_max,
                                                       efficiency_dict, error_rate, UMI_length)
        umis_in = post_cycle
    return umis_in


def addSequencingErrors(umi_counter, seq_error_rate=0.01):
    """
    takes a UMI counter object and adds errors at random to simulate sequencing errors

    :param umi_counter: Counter object representing pool of UMIs prepared for sequencing
    :param seq_error_rate: base calling error rate of the sequencing platform
    :return: Counter object representing pool of UMIs after sequencing
    """
    errors_dict = {"A": ["C", "G", "T"],
                   "C": ["A", "G", "T"],
                   "G": ["C", "A", "T"],
                   "T": ["C", "G", "A"]}
    new_list = []
    for umi in umi_counter.elements():
        new_umi = ""
        for base in umi:
            if np.random.random() > seq_error_rate:
                new_umi += base
            else:
                new_umi += np.random.choice(errors_dict[base])

        new_list.append(new_umi)

    return Counter(new_list)


def multiprocess_iterations(number_of_UMIs, length_of_UMIs, transcription_events, ligation_efficiency, RT_error_rate,
                            PCR_cycles,
                            t7_error_rate, error_rate, seq_error_rate, dilution_efficiency, efficiency_min,
                            efficiency_max):
    """
    Runs multiple iterations of the ST library preparation simulation in parallell using multiprocessing. Running iterations
    on multiple threads will greatly reduce the computation time.

    :param number_of_UMIs: staring number of UMIs used for simulation
    :param length_of_UMIs: number of bases in the UMI sequence
    :param transcription_events: average number of transcription events per molecule
    :param ligation_efficiency: proportion of UMIs left after ligation and purification steps before PCR amplification
    :param RT_error_rate: per base error rate of reverse transcriptase
    :param PCR_cycles: number of PCR cycles used for amplification
    :param t7_error_rate: per base error rate of T7 RNA polymerase
    :param error_rate: per base error rate of DNA polymerase used in the PCR reaction
    :param seq_error_rate: base calling error rate of the sequencing platform
    :param dilution_efficiency: proportion of UMIs left after dilution of amplified library
    :param efficiency_min: minimum efficiency of PCR amplification for one UMI sequence
    :param efficiency_max: maximum efficiency of PCR amplification for one UMI sequence
    :return: Counter object representing pool of UMIs after library preparation
    """
    np.random.seed()
    #UMIs, efficiency_dict = createTemplateUMIsWithBias(number_of_UMIs, length_of_UMIs,
    #                                           efficiency_min, efficiency_max)
    UMIs, efficiency_dict = createUMIsWithBias(number_of_UMIs, length_of_UMIs,
                                               efficiency_min, efficiency_max)
    transcribed_UMIs = TranscriptionEventsWithErrors(UMIs, t7_error_rate, transcription_events, length_of_UMIs)
    ligated_UMIs = downsampleUMIs(transcribed_UMIs, ligation_efficiency)
    reverse_transcribed_UMIs = transcribeUMIsWithErrors(ligated_UMIs, RT_error_rate, length_of_UMIs)
    final_UMIs = PCRcyclesWithErrorsAndBias(reverse_transcribed_UMIs, efficiency_min, efficiency_max,
                                            efficiency_dict, PCR_cycles, error_rate)
    sampled_UMIs = downsampleUMIs(final_UMIs, dilution_efficiency)
    UMIs_after_seq = addSequencingErrors(sampled_UMIs, seq_error_rate)
    return UMIs_after_seq


def CountsPerMethod(iterations, number_of_UMIs, length_of_UMIs, transcription_events, ligation_efficiency,
                    RT_error_rate, PCR_cycles,
                    T7_error_rate, error_rate, seq_error_rate, dilution_efficiency, efficiency_min, efficiency_max):
    no_dedup_counts = []
    unique_dedup_counts = []
    perc_dedup_counts = []
    cluster_dedup_counts = []
    adj_dedup_counts = []
    dir_adj_dedup_counts = []
    naive_dedup_counts = []
    hierarchical_single_counts = []
    hierarchical_complete_counts = []
    hierarchical_ward_counts = []
    affinity_counts = []

    pool = mp.Pool(processes=8)
    results = [pool.apply_async(multiprocess_iterations, args=(
    number_of_UMIs, length_of_UMIs, transcription_events, ligation_efficiency, RT_error_rate, PCR_cycles,
    T7_error_rate, error_rate, seq_error_rate, dilution_efficiency, efficiency_min, efficiency_max)) for i in
               range(iterations)]

    for p in results:

        UMIs_after_seq = p.get()
        naive_dedup_counts.append(dedup_naive(UMIs_after_seq))
        hierarchical_single_counts.append(hierarchical_single(UMIs_after_seq))
        hierarchical_complete_counts.append(hierarchical_complete(UMIs_after_seq))
        hierarchical_ward_counts.append(hierarchical_ward(UMIs_after_seq))
        if number_of_UMIs < 250:
            affinity_counts.append(dedup_dir_adj(UMIs_after_seq))
        else:
            affinity_counts.append(hierarchical_complete(UMIs_after_seq))
        no_dedup_counts.append(dedup_none(UMIs_after_seq))
        unique_dedup_counts.append(dedup_unique(UMIs_after_seq))
        perc_dedup_counts.append(dedup_percentile(UMIs_after_seq))
        cluster_dedup_counts.append(dedup_cluster(UMIs_after_seq))
        adj_dedup_counts.append(dedup_adj(UMIs_after_seq))
        dir_adj_dedup_counts.append(dedup_dir_adj(UMIs_after_seq))

    pool.terminate()
    print("Finished simulation of", number_of_UMIs, "UMIs")
    return (naive_dedup_counts, hierarchical_single_counts, hierarchical_complete_counts, hierarchical_ward_counts,
            affinity_counts, no_dedup_counts, unique_dedup_counts, perc_dedup_counts,
            cluster_dedup_counts, adj_dedup_counts, dir_adj_dedup_counts)


def simulationCycle(iterations, number_of_UMIs, length_of_UMIs, transcription_events, ligation_efficiency,
                    RT_error_rate, PCR_cycles, T7_error_rate, error_rate, seq_error_rate, dilution_efficiency,
                    efficiency_min, efficiency_max):
    dedup_methods = ["naive", "hier_simple", "hier_complete", "ward", "affinity",
                     "none", "unique", "perc", "cluster", "adj", "dir_adj"]

    counts = CountsPerMethod(iterations, number_of_UMIs, length_of_UMIs, transcription_events, ligation_efficiency,
                             RT_error_rate, PCR_cycles,
                             T7_error_rate, error_rate, seq_error_rate, dilution_efficiency, efficiency_min,
                             efficiency_max)

    (naive_dedup_counts, hierarchical_single_counts, hierarchical_complete_counts, hierarchical_ward_counts,
     affinity_counts, no_dedup_counts, unique_dedup_counts, perc_dedup_counts,
     cluster_dedup_counts, adj_dedup_counts, dir_adj_dedup_counts) = counts

    naive_dedup_CV = np.std(naive_dedup_counts) / np.mean(naive_dedup_counts)
    hierarchical_single_CV = np.std(hierarchical_single_counts) / np.mean(hierarchical_single_counts)
    hierarchical_complete_CV = np.std(hierarchical_complete_counts) / np.mean(hierarchical_complete_counts)
    hierarchical_ward_CV = np.std(hierarchical_ward_counts) / np.mean(hierarchical_ward_counts)
    affinity_CV = np.std(affinity_counts) / np.mean(affinity_counts)
    no_dedup_CV = np.std(no_dedup_counts) / np.mean(no_dedup_counts)
    unique_dedup_CV = np.std(unique_dedup_counts) / np.mean(unique_dedup_counts)
    perc_dedup_CV = np.std(perc_dedup_counts) / np.mean(perc_dedup_counts)
    cluster_dedup_CV = np.std(cluster_dedup_counts) / np.mean(cluster_dedup_counts)
    adj_dedup_CV = np.std(adj_dedup_counts) / np.mean(adj_dedup_counts)
    dir_adj_dedup_CV = np.std(dir_adj_dedup_counts) / np.mean(dir_adj_dedup_counts)

    CVs = naive_dedup_CV, hierarchical_single_CV, hierarchical_complete_CV, hierarchical_ward_CV, affinity_CV, no_dedup_CV, unique_dedup_CV, perc_dedup_CV, cluster_dedup_CV, adj_dedup_CV, dir_adj_dedup_CV
    tmp_df = pd.DataFrame({'number_of_molecules': (number_of_UMIs,) * len(dedup_methods),
                           'dedup': dedup_methods,
                           'count': [np.mean(x) for x in counts],
                           'CV': CVs})
    return tmp_df


def CV(counts):
    """
    calculate coefficient of variation

    :param counts: array of counts
    :return: CV value
    """
    return np.std(counts) / np.mean(counts)


class simulatorWithVariable(object):
    def __init__(self,
                 iterations=1000,
                 umi_length=7,
                 transcription_events=1000,
                 ligation_efficiency=0.4,
                 t7_error_rate=-4,
                 rt_error_rate=-4,
                 pcr_cycles=13,
                 dna_pol_error_rate=-5,
                 seq_error_rate=-3,
                 dilution_efficiency=2e-5,
                 eff_min=0.7,
                 eff_max=1.0,
                 number_of_umis=50):

        self.iterations = iterations
        self.umi_length = umi_length
        self.transcription_events = transcription_events
        self.ligation_efficiency = ligation_efficiency
        self.t7_error_rate = t7_error_rate
        self.rt_error_rate = rt_error_rate
        self.pcr_cycles = pcr_cycles
        self.dna_pol_error_rate = dna_pol_error_rate
        self.seq_error_rate = seq_error_rate
        self.dilution_efficiency = dilution_efficiency
        self.eff_min = eff_min
        self.eff_max = eff_max
        self.number_of_umis = number_of_umis
        self.variable = None

    def updateVariable(self, variable=None):

        if variable == "umi_length":
            self.umi_length = self.variable
        elif variable == "transcription_events":
            self.transcription_events = self.variable
        elif variable == "ligation_efficiency":
            self.ligation_efficiency = self.variable
        elif variable == "t7_error_rate":
            self.t7_error_rate = self.variable
        elif variable == "rt_error_rate":
            self.rt_error_rate = self.variable
        elif variable == "pcr_cycles":
            self.pcr_cycles = self.variable
        elif variable == "dna_pol_error_rate":
            self.dna_pol_error_rate = self.variable
        elif variable == "seq_error_rate":
            self.seq_error_rate = self.variable
        elif variable == "dilution_efficiency":
            self.dilution_efficiency = self.variable
        elif variable == "eff_min":
            self.eff_min = self.variable
        elif variable == "eff_max":
            self.eff_max = self.variable
        elif variable == "number_of_umis":
            self.number_of_umis = self.variable
        else:
            raise ValueError("Not a valid input parameter")

    def iterator(self, variable, iter_list):

        assert variable, "need to specify variable"

        dedup_methods = ["naive", "hier_simple", "hier_complete", "ward", "affinity",
                         "none", "unique", "perc", "cluster", "adj", "dir_adj"]

        df = pd.DataFrame()

        for value in iter_list:
            self.variable = value
            self.updateVariable(variable)

            t7_error_rate = 10 ** self.t7_error_rate
            rt_error_rate = 10 ** self.rt_error_rate
            pol_error_rate = 10 ** self.dna_pol_error_rate
            seq_error_rate = 10 ** self.seq_error_rate

            counts = CountsPerMethod(self.iterations,
                                     self.number_of_umis,
                                     self.umi_length,
                                     self.transcription_events,
                                     self.ligation_efficiency,
                                     rt_error_rate,
                                     self.pcr_cycles,
                                     t7_error_rate,
                                     pol_error_rate,
                                     seq_error_rate,
                                     self.dilution_efficiency,
                                     self.eff_min,
                                     self.eff_max)

            tmp_df = pd.DataFrame({"variable": (self.variable,) * len(dedup_methods),
                                   'dedup': dedup_methods,
                                   'count': [np.mean(x) for x in counts],
                                   'CV': [CV(x) for x in counts]})
            df = df.append(tmp_df)

        return df


def main():
    param_dict = dict()
    param_dict["transcription_events"] = np.arange(600, 2500, 100)
    param_dict["ligation_efficiency"] = np.arange(0.1, 0.84, 0.04)
    param_dict["t7_error_rate"] = np.arange(-6, -1.8, 0.2)
    param_dict["rt_error_rate"] = np.arange(-6, -1.8, 0.2)
    param_dict["pcr_cycles"] = np.arange(7, 18, 1)
    param_dict["dna_pol_error_rate"] = np.arange(-7, -2.8, 0.2)
    param_dict["seq_error_rate"] = np.arange(-5, -0.8, 0.2)
    param_dict["dilution_efficiency"] = np.arange(1e-5, 2.5e-4, 1e-5)
    param_dict["eff_min"] = np.arange(0.1, 1, 0.05)
    param_dict["number_of_umis"] = list(range(5, 85, 5)) + [100, 200, 300, 400, 500]
    for item in ["transcription_events", "ligation_efficiency", "t7_error_rate", "rt_error_rate", "pcr_cycles", "dna_pol_error_rate", "seq_error_rate", "dilution_efficiency", "eff_min", "number_of_umis"]:
        if item not in "number_of_umis":
            continue
        start = timeit.default_timer()
        iterations = 100
        #umi_list = list(range(5, 85, 5)) + [100, 200, 300, 400, 500]  # list(range(5, 85, 5)) +
        umi_list = param_dict[item]
        sim = simulatorWithVariable(iterations=iterations)
        df_Number_of_Molecules = sim.iterator(item, umi_list)
        print(df_Number_of_Molecules)
        stop = timeit.default_timer()
        print("Calculation time:", stop - start)
        df_Number_of_Molecules.to_csv("~/PycharmProjects/simulation_of_UMIs/CV_study_7_13c_{0}.csv".format(item), sep="\t")


if __name__ == "__main__":
    main()
