#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Jaromil Frossard <jaromil.frossard@gmail.com>
# the structure is copy-paste from cluster_level.py
# License: Simplified BSD
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
from mne.parallel import parallel_func, check_n_jobs
from mne.stats.cluster_level import (_check_fun,
                                     _setup_adjacency, _find_clusters, _cluster_indices_to_mask,
                                     _cluster_mask_to_indices, _get_partitions_from_adjacency)
from mne.utils import (verbose, split_list, ProgressBar, _check_option, _validate_type, check_random_state, logger, warn)


@verbose
def spatio_temporal_clusterdepth_test(
        X, threshold=None, n_permutations=1024, tail=0, stat_fun=None,
        n_jobs=1, seed=None,
        out_type='indices', exclude=None, verbose=None,
        check_disjoint=False, buffer_size=1000):
    # n_samples, n_times, n_vertices = X[0].shape
    # convert spatial_exclude before passing on if necessary
    # if spatial_exclude is not None:
    #    exclude = _st_mask_from_s_inds(n_times, n_vertices,
    #                                   spatial_exclude, True)
    # else:
    #    exclude = None
    return permutation_clusterdepth_test(
        X, threshold=threshold, stat_fun=stat_fun, tail=tail,
        n_permutations=n_permutations,
        n_jobs=n_jobs, seed=seed, buffer_size=buffer_size,
        out_type='indices', exclude=exclude, check_disjoint=check_disjoint,
        verbose=verbose)


@verbose
def permutation_clusterdepth_test(
        X, threshold=None, n_permutations=1024, tail=0, stat_fun=None,
        n_jobs=1, seed=None, exclude=None,
        out_type='indices',
        buffer_size=1000, check_disjoint=False, verbose=None):
    stat_fun, threshold = _check_fun(X, stat_fun, threshold, tail, 'between')
    return _permutation_clusterdepth_test(
        X=X, threshold=threshold, n_permutations=n_permutations, tail=tail,
        stat_fun=stat_fun, n_jobs=n_jobs, seed=seed,
        out_type='indices', exclude=exclude,
        check_disjoint=check_disjoint, buffer_size=buffer_size)


def _do_permutations_clusterdepth(X_full, slices, threshold, n_times, tail, adjacency, stat_fun,
                                  include, partitions, orders, sample_shape, buffer_size, progress_bar):
    n_samp, n_vars = X_full.shape

    if buffer_size is not None and n_vars <= buffer_size:
        buffer_size = None  # don't use buffer for few variables

    # allocate space for output
    clusterdepth_head = [[] for _ in range(len(orders))]
    clusterdepth_tail = [[] for _ in range(len(orders))]

    if buffer_size is not None:
        # allocate buffer, so we don't need to allocate memory during loop
        X_buffer = [np.empty((len(X_full[s]), buffer_size), dtype=X_full.dtype)
                    for s in slices]

    for seed_idx, order in enumerate(orders):
        # shuffle sample indices
        assert order is not None
        idx_shuffle_list = [order[s] for s in slices]

        if buffer_size is None:
            # shuffle all data at once
            X_shuffle_list = [X_full[idx, :] for idx in idx_shuffle_list]
            t_obs_surr = stat_fun(*X_shuffle_list)
        else:
            # only shuffle a small data buffer, so we need less memory
            t_obs_surr = np.empty(n_vars, dtype=X_full.dtype)

            for pos in range(0, n_vars, buffer_size):
                # number of variables for this loop
                n_var_loop = min(pos + buffer_size, n_vars) - pos

                # fill buffer
                for i, idx in enumerate(idx_shuffle_list):
                    X_buffer[i][:, :n_var_loop] = \
                        X_full[idx, pos: pos + n_var_loop]

                # apply stat_fun and store result
                tmp = stat_fun(*X_buffer)
                t_obs_surr[pos: pos + n_var_loop] = tmp[:n_var_loop]

        # The stat should have the same shape as the samples for no adj.
        if adjacency is None:
            t_obs_surr.shape = sample_shape

        # Find cluster on randomized stats
        out = _find_clusters(t_obs_surr, threshold, tail, adjacency,
                             max_step=1, include=include,
                             partitions=partitions, t_power=0,
                             show_info=True)

        progress_bar.update(seed_idx + 1)

        clusters, cluster_stats = out

        #cluster_stats = cluster_stats.astype(int)

        # identify which clusters is at the border
        starting = [cli[0] % n_times == 0 for cli in clusters]
        clh = [i for indx,i in enumerate(clusters) if not starting[indx]]

        ending = [cli[-1] + 1 % n_times == 0 for cli in clusters]
        clt = [i for indx,i in enumerate(clusters) if not ending[indx]]


        max_depth = max([len(cli) for cli in clusters], default=0)

        # cluster_stats = cluster_stats.astype(int)

        clusterdepth_head[seed_idx] = [0.0] * max_depth

        for depthi in range(max_depth):
            mx_cl = [0.] * len(clh)
            for cli in range(len(clh)):
                if len(clh[cli]) >= depthi + 1:
                    # reversing the clusters at the border
                    mx_cl[cli] = t_obs_surr[clh[cli][depthi]]
                else:
                    mx_cl[cli] = 0.0

            if tail == 1:
                clusterdepth_head[seed_idx][depthi] = max(mx_cl, default=0.0)
            elif tail == 0:
                clusterdepth_head[seed_idx][depthi] = max([abs(i) for i in mx_cl], default=0.0)

            elif tail == -1:
                clusterdepth_head[seed_idx][depthi] = min(mx_cl, default=0.0)
        # cdepth_tail = [0.] * max_depth

        clusterdepth_tail[seed_idx] = [0.0] * max_depth
        for depthi in range(max_depth):
            mx_cl = [0.0] * len(clt)
            for cli in range(len(clt)):
                if len(clt[cli]) >= depthi + 1:
                    mx_cl[cli] = t_obs_surr[clt[cli][-1 - depthi]]
                else:
                    mx_cl[cli] = 0.0
            if tail == 1:
                clusterdepth_tail[seed_idx][-1 - depthi] = max(mx_cl, default=0.0)
            elif tail == 0:
                clusterdepth_tail[seed_idx][-1 - depthi] = max([abs(i) for i in mx_cl], default=0.0)
            elif tail == -1:
                clusterdepth_tail[seed_idx][-1 - depthi] = min(mx_cl, default=0.0)

    #fill zero
    max_depth = max([len(i) for i in clusterdepth_head], default=0)
    clusterdepth_head = [i+([0.0]*(max_depth-len(i))) for i in clusterdepth_head]
    #clusterdepth_head = np.array(clusterdepth_head)

    max_depth = max([len(i) for i in clusterdepth_tail], default=0)
    clusterdepth_tail = [([0.0]*(max_depth-len(i)))+i for i in clusterdepth_tail]
    #clusterdepth_tail = np.array(clusterdepth_tail)
    return [clusterdepth_head, clusterdepth_tail]


def troendle(distribution, statistics, tail):
    if tail == 0:
        distribution = np.absolute(distribution)
        statistics = np.absolute(statistics)
    elif tail == -1:
        distribution = -distribution
        statistics = -statistics

    pos = np.concatenate((statistics, distribution))
    pos = np.apply_along_axis(lambda coli: stats.rankdata(coli, method="min"), 0, pos)
    pos = pos.shape[0] - pos + 1
    pvalues = np.array([np.nan] * pos.shape[1])
    test_order = sorted(np.unique(pos[0, :]))

    for testi in test_order:
        col_test = pos[0, :] == testi
        max_pval = 0
        if sum(col_test) > 0:
            col_distr = pos[0, :] >= testi
            distri = pos[:, col_distr]
            minp = np.apply_along_axis(min, 1, distri)
            pvali = np.mean(testi >= minp)
            # implement stopping rules by max
            pvali = max(pvali, max_pval)
            max_pval = pvali
            pvalues[col_test] = [pvali] * sum(col_test)
    return pvalues


def _permutation_clusterdepth_test(X, threshold, n_permutations, tail, stat_fun,
                                   n_jobs, seed, out_type, exclude,
                                   check_disjoint, buffer_size):
    n_jobs = check_n_jobs(n_jobs)
    """Aux Function.

    Note. X is required to be a list. Depending on the length of X
    either a 1 sample t-test or an F test / more sample permutation scheme
    is elicited.
    """
    _check_option('out_type', out_type, ['mask', 'indices'])
    _check_option('tail', tail, [-1, 0, 1])
    if not isinstance(threshold, dict):
        threshold = float(threshold)
        if (tail < 0 and threshold > 0 or tail > 0 and threshold < 0 or
                tail == 0 and threshold < 0):
            raise ValueError('incompatible tail and threshold signs, got '
                             '%s and %s' % (tail, threshold))

    # check dimensions for each group in X (a list at this stage).
    X = [x[:, np.newaxis] if x.ndim == 1 else x for x in X]
    n_samples = X[0].shape[0]
    n_times = X[0].shape[1]

    sample_shape = X[0].shape[1:]
    for x in X:
        if x.shape[1:] != sample_shape:
            raise ValueError('All samples mush have the same size')

    # flatten the last dimensions in case the data is high dimensional
    n_tests = np.prod(X[0].shape[1:])
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]

    adjacency = sparse.identity(np.prod(sample_shape[1:]))
    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)

    if (exclude is not None) and not exclude.size == n_tests:
        raise ValueError('exclude must be the same shape as X[0]')

    # Step 1: Calculate t-stat for original data
    # -------------------------------------------------------------
    t_obs = stat_fun(*X)
    _validate_type(t_obs, np.ndarray, 'return value of stat_fun')
    logger.info('stat_fun(H1): min=%f max=%f' % (np.min(t_obs), np.max(t_obs)))

    # test if stat_fun treats variables independently
    if buffer_size is not None:
        t_obs_buffer = np.zeros_like(t_obs)
        for pos in range(0, n_tests, buffer_size):
            t_obs_buffer[pos: pos + buffer_size] = \
                stat_fun(*[x[:, pos: pos + buffer_size] for x in X])

        if not np.alltrue(t_obs == t_obs_buffer):
            warn('Provided stat_fun does not treat variables independently. '
                 'Setting buffer_size to None.')
            buffer_size = None

    # The stat should have the same shape as the samples for no adj.
    if t_obs.size != np.prod(sample_shape):
        raise ValueError('t_obs.shape %s provided by stat_fun %s is not '
                         'compatible with the sample shape %s'
                         % (t_obs.shape, stat_fun, sample_shape))
    if adjacency is None or adjacency is False:
        t_obs.shape = sample_shape

    if exclude is not None:
        include = np.logical_not(exclude)
    else:
        include = None

    # determine if adjacency itself can be separated into disjoint sets
    if check_disjoint is True and (adjacency is not None and
                                   adjacency is not False):
        partitions = _get_partitions_from_adjacency(adjacency, n_times)
    else:
        partitions = None
    logger.info('Running initial clustering')
    out = _find_clusters(t_obs, threshold, tail, adjacency,
                         max_step=1, include=include,
                         partitions=partitions, t_power=0,
                         show_info=True)
    clusters, cluster_length = out

    starting = [cli[0] % n_times == 0 for cli in clusters]
    ending = [cli[-1] + 1 % n_times == 0 for cli in clusters]

    clusters = [i for indx, i in enumerate(clusters) if not starting[indx] and not ending[indx]]
    cluster_length = [i for indx, i in enumerate(cluster_length) if not starting[indx] and not ending[indx]]



    cluster_length = [int(li) for li in cluster_length]

    max_depth = max(cluster_length, default=0)

    logger.info('Found %d clusters' % len(clusters))

    # convert clusters to old format
    if adjacency is not None and adjacency is not False:
        # our algorithms output lists of indices by default
        if out_type == 'mask':
            clusters = _cluster_indices_to_mask(clusters, n_tests)
    else:
        # ndimage outputs slices or boolean masks by default
        if out_type == 'indices':
            clusters = _cluster_mask_to_indices(clusters, t_obs.shape)

    # convert our seed to orders
    # check to see if we can do an exact test
    # (for a two-tailed test, we can exploit symmetry to just do half)
    extra = ''
    rng = check_random_state(seed)
    del seed
    # if len(X) == 1:  # 1-sample test
    # do_perm_func = _do_1samp_permutations
    # X_full = X[0]
    # slices = None
    # orders, n_permutations, extra = _get_1samp_orders(
    #    n_samples, n_permutations, tail, rng)
    # else:
    n_permutations = int(n_permutations)
    do_perm_func = _do_permutations_clusterdepth
    X_full = np.concatenate(X, axis=0)
    n_samples_per_condition = [x.shape[0] for x in X]
    splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
    slices = [slice(splits_idx[k], splits_idx[k + 1])
              for k in range(len(X))]
    orders = [rng.permutation(len(X_full))
              for _ in range(n_permutations - 1)]
    del rng
    parallel, my_do_perm_func, _ = parallel_func(
        do_perm_func, n_jobs, verbose=False)

    if len(clusters) == 0:
        warn('No clusters found, returning empty H0, clusters, and cluster_pv')
        return t_obs, np.array([]), np.array([]), np.array([])

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------
    # Step 3: repeat permutations for step-down-in-jumps procedure
    total_removed = 0
    step_down_include = None  # start out including all points
    n_step_downs = 0

    # while n_removed > 0:
    # actually do the clustering for each partition
    if include is not None:
        if step_down_include is not None:
            this_include = np.logical_and(include, step_down_include)
        else:
            this_include = include
    else:
        this_include = step_down_include
    logger.info('Permuting %d times%s...' % (len(orders), extra))

    with ProgressBar(len(orders)) as progress_bar:
        H0 = parallel(
            my_do_perm_func(X_full, slices, threshold, n_times, tail, adjacency,
                            stat_fun, this_include, partitions,
                            order, sample_shape, buffer_size,
                            progress_bar.subset(idx))
            for idx, order in split_list(orders, n_jobs, idx=True))
    logger.info('Done.')

    clusterdepth_head = np.array(H0[0][0])
    clusterdepth_tail = np.array(H0[0][1])

    pvalues = [None] * len(clusters)
    for i in range(len(clusters)):
        nfill = clusterdepth_head.shape[1]-len(clusters[i])
        statistics = np.concatenate((t_obs[clusters[i]], np.zeros(nfill)))
        statistics.shape = (1, statistics.shape[0])
        pvalues_head = troendle(clusterdepth_head, statistics, tail=tail)
        pvalues_head = pvalues_head[range(len(clusters[i]))]

        nfill = clusterdepth_tail.shape[1] - len(clusters[i])
        statistics = np.concatenate((np.zeros(nfill), t_obs[clusters[i]]))
        statistics.shape = (1, statistics.shape[0])
        pvalues_tail = troendle(clusterdepth_tail, statistics, tail=tail)
        pvalues_tail = pvalues_tail[range(nfill, len(pvalues_tail))]

        pvalues[i] = np.maximum(pvalues_head, pvalues_tail)

    # clusters = _reshape_clusters(clusters, sample_shape)
    return t_obs, clusters, pvalues, [clusterdepth_head, clusterdepth_tail]
