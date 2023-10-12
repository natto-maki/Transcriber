import sys
import os
import pickle
import time
import datetime
import random
import threading as th
import dataclasses
import logging
import json
import colorsys
import argparse
import copy

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import japanize_matplotlib

import tools
import measure_time

japanize_matplotlib.japanize()


@dataclasses.dataclass
class Person:
    person_id: int = -1
    superseded_by: int = -1
    name: str = ""
    is_default: bool = False
    # Time when the new embedding was map()'ed; <0 means not mapped to specific cluster
    last_mapped_time: float = -1.0


_update_interval = 8


def _dump_embeddings(target, shape_only):
    if not shape_only:
        return target
    if target is None:
        return None
    if isinstance(target, np.ndarray):
        return "ndarray(shape=%s)" % (target.shape,)
    if isinstance(target, list):
        if len(target) == 0:
            return []
        if isinstance(target[0], np.ndarray) and len(target[0].shape) == 1:
            return "list[%d ndarray(shape=(%d,))]" % (len(target), target[0].shape[0])
    raise ValueError()


def _embedding_hash(x):
    return np.average(x, axis=1)


def _cosine_distance_mat(emb):
    n = 1 / np.fmax(sys.float_info.epsilon, np.linalg.norm(emb, axis=1))
    return np.fmax(0.0, 1.0 - (emb @ np.transpose(emb)) * n * np.reshape(n, [-1, 1]))


@dataclasses.dataclass
class _Cluster:
    cluster_id: int = -1
    linked_person_ids: list[int] = dataclasses.field(default_factory=list)
    core_embeddings: np.ndarray | None = None
    sub_embeddings: np.ndarray | None = None
    last_mapped_time: float = 0.0  # Time when the new embedding was map()'ed

    def dump(self, shape_only):
        return {
            "cluster_id": self.cluster_id,
            "linked_person_ids": copy.copy(self.linked_person_ids),
            "core_embeddings": _dump_embeddings(self.core_embeddings, shape_only),
            "sub_embeddings": _dump_embeddings(self.sub_embeddings, shape_only),
            "last_mapped_time": time.ctime(self.last_mapped_time)
        }


@dataclasses.dataclass
class _Person:
    person_id: int = -1
    cluster_id: int = -1
    superseded_by: int = -1  # person_id; if another _Person points to _Cluster
    core_embeddings: np.ndarray | None = None
    name: str = ""
    name_history: list[str] = dataclasses.field(default_factory=list)
    default_name: str = ""
    last_mapped_time: float = 0.0  # Time last mapped to _Cluster
    last_updated_time: float = 0.0  # Time when the name was updated

    def dump(self, shape_only):
        return {
            "person_id": self.person_id,
            "cluster_id": self.cluster_id,
            "superseded_by": self.superseded_by,
            "core_embeddings": _dump_embeddings(self.core_embeddings, shape_only),
            "name": self.name,
            "name_history": copy.copy(self.name_history),
            "default_name": self.default_name,
            "last_mapped_time": time.ctime(self.last_mapped_time),
            "last_updated_time": time.ctime(self.last_updated_time)
        }


class EmbeddingDatabase:
    def __init__(self, embedding_dim: int, database_file_name: str | None = None,
                 threshold=0.7, dbscan_eps=0.65, dbscan_min_samples=3,
                 min_matched_embeddings_to_inherit_cluster=3,
                 min_matched_embeddings_to_match_person=3,
                 reduce_embeddings_threshold_size=0,
                 preferred_cluster_size=100, preferred_cluster_size_scale=0.75, distance_threshold_for_cluster=0.1,
                 preferred_person_size=100, distance_threshold_for_person=0.1):

        self.__embedding_dim = embedding_dim
        self.__database_file_name = database_file_name
        self.__threshold = threshold
        self.__dbscan_eps = dbscan_eps
        self.__dbscan_min_samples = dbscan_min_samples
        self.__min_matched_embeddings_to_inherit_cluster = min_matched_embeddings_to_inherit_cluster
        self.__min_matched_embeddings_to_match_person = min_matched_embeddings_to_match_person
        self.__reduce_embeddings_threshold_size = reduce_embeddings_threshold_size
        self.__preferred_cluster_size = preferred_cluster_size
        self.__preferred_cluster_size_scale = preferred_cluster_size_scale
        self.__distance_threshold_for_cluster = distance_threshold_for_cluster
        self.__preferred_person_size = preferred_person_size
        self.__distance_threshold_for_person = distance_threshold_for_person

        self.__lock0 = th.Lock()
        self.__clusters: dict[int, _Cluster] = {}
        self.__persons: dict[int, _Person] = {}
        self.__unprocessed_embeddings: list[np.ndarray] = []
        self.__unassigned_embeddings: np.ndarray | None = None
        self.__next_cluster_index = 1
        self.__next_person_index = 1
        self.__generation = int(time.time())
        self.__mapped_to_unknown_count = 0

        if database_file_name is not None:
            tools.recover_file(database_file_name)
            if os.path.isfile(database_file_name):
                with open(database_file_name, "rb") as f:
                    d = pickle.load(f)
                self.__clusters = d["clusters"]
                self.__persons = d["persons"]
                self.__unprocessed_embeddings = d["unprocessed_embeddings"]
                self.__unassigned_embeddings = d["unassigned_embeddings"]
                self.__next_cluster_index = d["next_cluster_index"]
                self.__next_person_index = d["next_person_index"]
                self.__generation = d["generation"]

        self.__cache_core_x: np.ndarray | None = None
        self.__cache_core_y: np.ndarray | None = None
        self.__cache_core_embedding_hash: dict[int, np.ndarray] = {}

        self.__update_cache()

        with open("resources/name_us.txt", "r") as f:
            self.__placeholder_names = f.read().split('\n')
            random.shuffle(self.__placeholder_names)

    @staticmethod
    def metrics(emb0, emb1):
        return max(0.0, 1.0 - np.dot(emb0, emb1) / (np.linalg.norm(emb0) * np.linalg.norm(emb1)))

    @staticmethod
    def __metrics_mat(emb):
        return _cosine_distance_mat(emb)

    @staticmethod
    def __find_nearest(target, emb):
        if len(emb) == 0:
            raise ValueError
        if len(target) == 0:
            return np.full((len(emb),), -1), np.full((len(emb),), np.finfo(np.float32).max)
        n0 = 1 / np.fmax(sys.float_info.epsilon, np.linalg.norm(target, axis=1))
        n1 = 1 / np.fmax(sys.float_info.epsilon, np.linalg.norm(emb, axis=1))
        d = np.fmax(0.0, 1.0 - (emb @ np.transpose(target)) * n0 * np.reshape(n1, [-1, 1]))
        return np.argmin(d, axis=1), np.min(d, axis=1)

    def __add_new_person(self, updated_time: float, core_embeddings: np.ndarray):
        p = _Person()
        p.person_id = self.__next_person_index
        self.__next_person_index += 1
        p.core_embeddings = core_embeddings
        p.default_name = "person_%d" % p.person_id
        if len(self.__placeholder_names) != 0:
            p.default_name = self.__placeholder_names[-1]
            del self.__placeholder_names[-1]
        p.name = p.default_name
        p.name_history.append(p.default_name)
        p.last_updated_time = updated_time
        self.__persons[p.person_id] = p
        return p

    def __update_cache(self):
        emb_count = 0
        for c in self.__clusters.values():
            emb_count += len(c.core_embeddings)

        self.__cache_core_x = np.zeros((emb_count, self.__embedding_dim), dtype=np.float32)
        self.__cache_core_y = np.zeros((emb_count,), dtype=np.int32)

        offset = 0

        def _push_embeddings(index_, embeddings_):
            nonlocal offset
            length_ = len(embeddings_)
            self.__cache_core_x[offset:offset + length_] = embeddings_
            self.__cache_core_y[offset:offset + length_] = index_
            offset += length_

        for c in self.__clusters.values():
            _push_embeddings(c.cluster_id, c.core_embeddings)

        assert offset == emb_count

    def __count_all_embeddings(self):
        emb_count = len(self.__unprocessed_embeddings)
        if self.__unassigned_embeddings is not None:
            emb_count += len(self.__unassigned_embeddings)
        for c in self.__clusters.values():
            emb_count += len(c.core_embeddings)
            if c.sub_embeddings is not None:
                emb_count += len(c.sub_embeddings)
        return emb_count

    def __count_all_person_embeddings(self):
        emb_count = 0
        for p in self.__persons.values():
            emb_count += len(p.core_embeddings)
        return emb_count

    def __stack_all_embeddings(self, callback, include_person=False):
        offset = 0

        def _push_embeddings(embeddings_, index_, is_core_):
            nonlocal offset
            length_ = len(embeddings_)
            callback(offset, length_, embeddings_, index_, is_core_)
            offset += length_

        if len(self.__unprocessed_embeddings) != 0:
            _push_embeddings(np.vstack(self.__unprocessed_embeddings), -2, False)
        if self.__unassigned_embeddings is not None:
            _push_embeddings(self.__unassigned_embeddings, -1, False)
        for c in self.__clusters.values():
            _push_embeddings(c.core_embeddings, c.cluster_id, True)
            if c.sub_embeddings is not None:
                _push_embeddings(c.sub_embeddings, c.cluster_id, False)

        if include_person:
            for p in self.__persons.values():
                _push_embeddings(p.core_embeddings, -3, True)

    def __get_all_embeddings_x(self, include_person=False):
        emb_count = self.__count_all_embeddings()
        if include_person:
            emb_count += self.__count_all_person_embeddings()

        x0 = np.zeros((emb_count, self.__embedding_dim), dtype=np.float32)

        def _push_embeddings(offset_, length_, embeddings_, index_, is_core_):
            nonlocal x0
            x0[offset_:offset_ + length_] = embeddings_
            _ = index_
            _ = is_core_

        self.__stack_all_embeddings(_push_embeddings, include_person=include_person)
        return x0

    def __get_all_embeddings_full(self):
        emb_count = self.__count_all_embeddings()

        x0 = np.zeros((emb_count, self.__embedding_dim), dtype=np.float32)
        y0 = np.zeros((emb_count,), dtype=np.int32)
        c0 = np.zeros((emb_count,), dtype=bool)

        def _push_embeddings(offset_, length_, embeddings_, index_, is_core_):
            nonlocal x0, y0, c0
            x0[offset_:offset_ + length_] = embeddings_
            y0[offset_:offset_ + length_] = index_
            c0[offset_:offset_ + length_] = is_core_

        self.__stack_all_embeddings(_push_embeddings)
        return x0, y0, c0

    def __get_all_embeddings_for_mapping(self):
        emb_count0 = self.__count_all_embeddings()
        emb_count1 = self.__count_all_person_embeddings()

        x0 = np.zeros((emb_count0 + emb_count1, self.__embedding_dim), dtype=np.float32)

        def _push_embeddings(offset_, length_, embeddings_, index_, is_core_):
            nonlocal x0
            x0[offset_:offset_ + length_] = embeddings_
            _ = index_
            _ = is_core_

        self.__stack_all_embeddings(_push_embeddings, include_person=True)
        return x0, emb_count0

    def __update_cache_core_embedding_hash(self):
        self.__cache_core_embedding_hash.clear()
        for c in self.__clusters.values():
            self.__cache_core_embedding_hash[c.cluster_id] = _embedding_hash(c.core_embeddings)

    def __find_matched_cluster(self, embeddings_hash, min_matched_embeddings=1):
        max_matched_count = 0
        max_matched_cluster_id = -1
        for c in self.__clusters.values():
            matched_count = len(np.intersect1d(self.__cache_core_embedding_hash[c.cluster_id], embeddings_hash))
            if min_matched_embeddings <= matched_count and max_matched_count < matched_count:
                max_matched_count = matched_count
                max_matched_cluster_id = c.cluster_id
        return max_matched_cluster_id, max_matched_count

    def __find_core_embedding_clusters(self, embeddings, target_size=-1, target_cluster_size=-1):
        assert target_size != -1 or target_cluster_size != -1

        if len(embeddings) <= target_cluster_size:
            return None
        distances = self.__metrics_mat(embeddings)
        result = None
        eps_scale_log2 = 0.0

        for _ in range(20):
            eps_scale_log2 -= 0.05
            clustering = DBSCAN(
                min_samples=self.__dbscan_min_samples, metric="precomputed",
                eps=self.__dbscan_eps * (2.0 ** eps_scale_log2))
            y0 = clustering.fit_predict(distances)
            cluster_count = int(np.max(y0)) + 1

            # logging.info(
            #     "find_core_embedding_clusters: eps_scale_log2 = %f, eps = %f, cluster_count = %d, clustered = %d" % (
            #     eps_scale_log2, self.__dbscan_eps * (2.0 ** eps_scale_log2), int(np.max(y0)) + 1, len(y0[y0 != -1])))

            if cluster_count == 0:
                return result

            result = y0

            if target_size != -1:
                if np.count_nonzero(y0 != -1) <= target_size:
                    return y0

            if target_cluster_size != -1:
                min_cluster_size = sys.maxsize
                for i in range(cluster_count):
                    min_cluster_size = min(min_cluster_size, np.count_nonzero(y0 == i))
                assert min_cluster_size != 0
                if min_cluster_size <= target_cluster_size:
                    return y0
        else:
            return None

    def __remove_neighbor_embeddings(
            self, embeddings, distance_threshold, size_lower_bound=0, removed_embeddings_store: list | None = None):
        size = len(embeddings)
        if size < size_lower_bound:
            return embeddings
        x0 = embeddings.copy()
        np.random.shuffle(x0)
        mask = ((self.__metrics_mat(x0) + np.tril(np.ones((size, size))).T).min(axis=1) > distance_threshold)
        masked = x0[mask]
        if len(masked) < size_lower_bound:
            return embeddings
        if removed_embeddings_store is not None:
            removed_embeddings_store.append(x0[np.logical_not(mask)])

        # logging.info("remove_neighbor_embeddings: %d -> %d (%d)" % (size, len(masked), len(masked) - size))

        return masked

    def __reduce_cluster_embeddings(
            self, preferred_cluster_size=100, preferred_cluster_size_scale=0.75, distance_threshold_for_cluster=0.1,
            removed_embeddings_store: list | None = None):

        kwargs_shrink_cluster = {
            "distance_threshold": distance_threshold_for_cluster,
            "size_lower_bound": preferred_cluster_size,
            "removed_embeddings_store": removed_embeddings_store
        }
        for c in self.__clusters.values():
            c.core_embeddings = self.__remove_neighbor_embeddings(c.core_embeddings, **kwargs_shrink_cluster)
            if c.sub_embeddings is not None:
                c.sub_embeddings = self.__remove_neighbor_embeddings(c.sub_embeddings, **kwargs_shrink_cluster)

            size_c = len(c.core_embeddings)
            size_s = len(c.sub_embeddings) if c.sub_embeddings is not None else 0
            if size_c + size_s <= preferred_cluster_size:
                continue

            target_size = int(preferred_cluster_size * np.exp(np.log(
                (size_c + size_s) / preferred_cluster_size) * preferred_cluster_size_scale))
            embeddings = np.concatenate([c.core_embeddings, c.sub_embeddings]) \
                if c.sub_embeddings is not None else c.core_embeddings
            clusters = self.__find_core_embedding_clusters(embeddings, target_size=target_size)

            if clusters is None or np.count_nonzero(clusters[0:size_c] != -1) < self.__dbscan_min_samples:
                continue

            if removed_embeddings_store is not None:
                removed_embeddings_store.append(embeddings[clusters == -1])

            mask = (clusters != -1)
            c.core_embeddings = embeddings[:size_c][mask[:size_c]]
            c.sub_embeddings = embeddings[size_c:][mask[size_c:]] if np.count_nonzero(mask[size_c:]) != 0 else None

            logging.info("cluster %d (%s): target %d; reduced %d(%d + %d) -> %d(%d + %d)" % (
                c.cluster_id,
                self.__persons[c.linked_person_ids[0]].name
                if len(c.linked_person_ids) != 0 and c.linked_person_ids[0] in self.__persons else "unmapped",
                target_size, size_c + size_s, size_c, size_s,
                np.count_nonzero(mask), np.count_nonzero(mask[:size_c]), np.count_nonzero(mask[size_c:])))

    def __reduce_person_embeddings(
            self, preferred_cluster_size=100, distance_threshold_for_person=0.1,
            removed_embeddings_store: list | None = None):

        for p in self.__persons.values():
            p.core_embeddings = self.__remove_neighbor_embeddings(
                p.core_embeddings,
                distance_threshold=distance_threshold_for_person,
                size_lower_bound=preferred_cluster_size,
                removed_embeddings_store=removed_embeddings_store)

        original_person_id_map = {p.person_id: p.person_id for p in self.__persons.values()}

        required_next_update = True
        while required_next_update:
            required_next_update = False
            divided_persons = []
            for p in self.__persons.values():
                y0 = self.__find_core_embedding_clusters(
                    p.core_embeddings, target_cluster_size=preferred_cluster_size)
                if y0 is None:
                    continue

                cluster_count = int(np.max(y0)) + 1
                if cluster_count == 1:
                    p.core_embeddings = p.core_embeddings[y0 != -1]
                    continue

                required_next_update = True

                original_embeddings = p.core_embeddings
                p.core_embeddings = original_embeddings[y0 == 0]
                for i in range(1, cluster_count):
                    p_sub = copy.deepcopy(p)
                    p_sub.core_embeddings = original_embeddings[y0 == i]
                    p_sub.person_id = self.__next_person_index
                    self.__next_person_index += 1
                    divided_persons.append(p_sub)
                    original_person_id_map[p_sub.person_id] = original_person_id_map[p.person_id]

            for p in divided_persons:
                self.__persons[p.person_id] = p

        return original_person_id_map

    def __reunion_person_embeddings(self, original_person_id_map):
        for c in self.__clusters.values():
            index = 0
            while index + 1 < len(c.linked_person_ids):
                original_person_id = original_person_id_map[c.linked_person_ids[index]]
                for index_delta, person_id in enumerate(c.linked_person_ids[index + 1:]):
                    if original_person_id != original_person_id_map[person_id]:
                        continue
                    target_person_id = c.linked_person_ids[index]
                    if person_id == original_person_id:
                        target_person_id, person_id = person_id, target_person_id
                    p = self.__persons[target_person_id]
                    p.core_embeddings = np.concatenate([p.core_embeddings, self.__persons[person_id].core_embeddings])
                    c.linked_person_ids[index] = target_person_id
                    del c.linked_person_ids[index + 1 + index_delta]
                    del self.__persons[person_id]
                    break
                else:
                    index += 1

    def __reconstruct(self, force_reduce=False):
        tm0 = time.time()

        with measure_time.Measure("__reconstruct.prepare"):
            reduce_embeddings = (force_reduce or (
                self.__reduce_embeddings_threshold_size != 0 and
                self.__count_all_embeddings() + self.__count_all_person_embeddings()
                > self.__reduce_embeddings_threshold_size))

            original_person_id_map = {}
            removed_embeddings_store = []
            if reduce_embeddings:
                self.__reduce_cluster_embeddings(
                    self.__preferred_cluster_size, self.__preferred_cluster_size_scale,
                    self.__distance_threshold_for_cluster, removed_embeddings_store)
                original_person_id_map = self.__reduce_person_embeddings(
                    self.__preferred_person_size, self.__distance_threshold_for_person, removed_embeddings_store)

            x0 = self.__get_all_embeddings_x(include_person=reduce_embeddings)
            if len(x0) == 0:
                return
            self.__unprocessed_embeddings.clear()
            self.__mapped_to_unknown_count = 0

        with measure_time.Measure("__reconstruct.clustering"):
            # TODO Add a constant to the distance between points of _Person origin
            #  that the user explicitly treats as separate persons.
            clustering = DBSCAN(eps=self.__dbscan_eps, min_samples=self.__dbscan_min_samples, metric="precomputed")
            y0 = clustering.fit_predict(self.__metrics_mat(x0))
            cluster_count = int(np.max(y0)) + 1

        # Re-map the clusters
        # Check if the clusters can be mapped to an existing _Cluster
        # (Multiple clusters can be mapped to a single _Cluster)
        if reduce_embeddings:
            self.__clusters.clear()
            cluster_mapping = [-1 for _ in range(cluster_count)]
        else:
            with measure_time.Measure("__reconstruct.map_clusters"):
                self.__update_cache_core_embedding_hash()
                x0h = _embedding_hash(x0)
                cluster_mapping = []  # cluster_id or -1 if not mapped
                for i in range(cluster_count):
                    matched_cluster_id, _ = self.__find_matched_cluster(
                        x0h[y0 == i], self.__min_matched_embeddings_to_inherit_cluster)
                    cluster_mapping.append(matched_cluster_id)

            with measure_time.Measure("__reconstruct.migrate_clusters"):
                cluster_id_to_remove = []
                for c in self.__clusters.values():
                    c.linked_person_ids.clear()
                    c.sub_embeddings = None

                    mask = np.zeros((len(x0),), dtype=bool)
                    for i, cluster_id in enumerate(cluster_mapping):
                        if cluster_id == c.cluster_id:
                            mask += (y0 == i)
                    if not np.max(mask):
                        cluster_id_to_remove.append(c.cluster_id)
                    else:
                        c.core_embeddings = x0[mask]

                # Delete _Cluster where embedding was not mapped
                for cluster_id in cluster_id_to_remove:
                    del self.__clusters[cluster_id]

        # Clusters not mapped to an existing _Cluster are created as a new _Cluster
        with measure_time.Measure("__reconstruct.add_new_clusters"):
            for i, cluster_id in enumerate(cluster_mapping):
                if cluster_id != -1:
                    continue

                c = _Cluster()
                c.cluster_id = self.__next_cluster_index
                self.__next_cluster_index += 1
                c.core_embeddings = x0[y0 == i]
                self.__clusters[c.cluster_id] = c

                cluster_mapping[i] = c.cluster_id

        # Update sub_embeddings
        with measure_time.Measure("__reconstruct.update_sub_embeddings"):
            x1 = x0[y0 != -1]
            y1 = y0[y0 != -1]
            x2 = x0[y0 == -1]
            if len(x2) == 0:
                self.__unassigned_embeddings = None
            else:
                nearest_index, nearest_distance = self.__find_nearest(x1, x2)
                assigned_mask = np.zeros((len(x2), ), dtype=np.int32)
                for i, cluster_id in enumerate(cluster_mapping):
                    assert cluster_id != -1
                    mask = (y1[nearest_index] == i) * (nearest_distance < self.__threshold)
                    if not np.max(mask):
                        continue
                    c = self.__clusters[cluster_id]
                    if c.sub_embeddings is None:
                        c.sub_embeddings = x2[mask]
                    else:
                        c.sub_embeddings = np.vstack([c.sub_embeddings, x2[mask]])
                    assigned_mask += mask.astype(np.int32)

                self.__unassigned_embeddings = x2[assigned_mask == 0]

        # Re-map the persons
        with measure_time.Measure("__reconstruct.map_persons"):
            self.__update_cache_core_embedding_hash()
            for p in self.__persons.values():
                p.cluster_id = -1
                p.superseded_by = -1
                matched_cluster_id, _ = self.__find_matched_cluster(
                    _embedding_hash(p.core_embeddings), self.__min_matched_embeddings_to_match_person)
                if matched_cluster_id != -1:
                    p.cluster_id = matched_cluster_id
                    p.last_mapped_time = tm0
                    self.__clusters[p.cluster_id].linked_person_ids.append(p.person_id)

            if reduce_embeddings:
                self.__reunion_person_embeddings(original_person_id_map)

        with measure_time.Measure("__reconstruct.migrate_persons"):
            person_id_to_remove = []

            def _person_score(p_):
                return -(p_.last_updated_time + (tm0 if p_.name != p_.default_name else 0.0))

            for c in self.__clusters.values():
                if len(c.linked_person_ids) < 2:
                    continue

                c.linked_person_ids.sort(key=lambda person_id_: _person_score(self.__persons[person_id_]))
                p0 = self.__persons[c.linked_person_ids[0]]
                p0_is_named = (p0.name != p0.default_name)
                for person_id in c.linked_person_ids[1:]:
                    p = self.__persons[person_id]
                    p.superseded_by = c.linked_person_ids[0]
                    if p0_is_named and p.name == p.default_name:
                        person_id_to_remove.append(person_id)

            for person_id in person_id_to_remove:
                del self.__persons[person_id]

        # Assign a _Person to a _Cluster that does not have an existing _Person mapped to it
        for c in self.__clusters.values():
            if len(c.linked_person_ids) == 0:
                p = self.__add_new_person(tm0, c.core_embeddings)
                p.cluster_id = c.cluster_id
                c.linked_person_ids.append(p.person_id)

        self.__generation += 1
        self.__update_cache()

        return np.concatenate(removed_embeddings_store) if len(removed_embeddings_store) != 0 else None

    def __map(self, tm0: float, embeddings: np.ndarray) -> list[int]:
        ret = []
        nearest_index, nearest_distance = self.__find_nearest(self.__cache_core_x, embeddings)
        for i in range(len(embeddings)):
            if nearest_distance[i] < self.__threshold:
                c = self.__clusters[self.__cache_core_y[nearest_index[i]]]
                c.last_mapped_time = tm0
                assert len(c.linked_person_ids) != 0
                ret.append(c.linked_person_ids[0])
            else:
                ret.append(-1)
        return ret

    def map(self, embeddings: list[np.ndarray] | np.ndarray, update=True) -> list[[int, str | None]]:
        if isinstance(embeddings, list):
            embeddings_l = embeddings
            embeddings_np = np.vstack(embeddings)
        elif isinstance(embeddings, np.ndarray) and \
                len(embeddings.shape) == 2 and embeddings.shape[1] == self.__embedding_dim:
            embeddings_l = [embeddings[i] for i in range(embeddings.shape[0])]
            embeddings_np = embeddings
        else:
            raise ValueError

        tm0 = time.time()
        ret = []
        with self.__lock0:
            if update:
                self.__unprocessed_embeddings.extend(embeddings_l)

            mapped = self.__map(tm0, embeddings_np)
            for person_id in mapped:
                if person_id != -1:
                    p = self.__persons[person_id]
                    ret.append([person_id, p.name])
                else:
                    ret.append([-1, None])
                    if update:
                        self.__mapped_to_unknown_count += 1

            if update and self.__mapped_to_unknown_count >= _update_interval:
                self.__reconstruct()

        return ret

    def get_persons(self) -> list[Person]:
        ret = []
        with self.__lock0:
            for p in self.__persons.values():
                ret.append(Person(
                    person_id=p.person_id, superseded_by=p.superseded_by,
                    name=p.name, is_default=(p.name == p.default_name),
                    last_mapped_time=self.__clusters[p.cluster_id].last_mapped_time if p.cluster_id != -1 else -1.0))
        return ret

    def add_person(self, core_embeddings: np.array, name: str):
        with self.__lock0:
            p = self.__add_new_person(time.time(), core_embeddings)
            p.name = name
            p.name_history.append(name)
            p.last_updated_time = time.time()

            self.__reconstruct()

    def rename(self, person_id: int, new_name: str):
        if len(new_name) == 0:
            return
        with self.__lock0:
            self.__reconstruct()

            if person_id not in self.__persons:
                return
            p = self.__persons[person_id]
            if p.name == new_name:
                return
            p.name = new_name
            p.name_history.append(new_name)
            p.last_updated_time = time.time()

            self.__reconstruct()

    def erase(self, person_id: int):
        with self.__lock0:
            if person_id not in self.__persons:
                return
            del self.__persons[person_id]
            self.__reconstruct()

    def erase_all(self):
        with self.__lock0:
            self.__persons = {}
            self.__reconstruct()

    def sync(self, file_name=None):
        if file_name is None:
            file_name = self.__database_file_name
        if file_name is None:
            return
        with self.__lock0:
            with tools.SafeWrite(file_name, "wb") as f:
                pickle.dump({
                    "clusters": self.__clusters,
                    "persons": self.__persons,
                    "unprocessed_embeddings": self.__unprocessed_embeddings,
                    "unassigned_embeddings": self.__unassigned_embeddings,
                    "next_cluster_index": self.__next_cluster_index,
                    "next_person_index": self.__next_person_index,
                    "generation": self.__generation
                }, f.stream)

    def reconstruct(self, force_reconstruct=False, force_reduce=False):
        with self.__lock0:
            if force_reconstruct or self.__mapped_to_unknown_count != 0:
                return self.__reconstruct(force_reduce=force_reduce)
        return None

    def get_generation(self):
        with self.__lock0:
            return self.__generation

    def __dump_state(self, shape_only):
        x0, y0, c0 = self.__get_all_embeddings_full()
        return {
            "x": _dump_embeddings(x0, shape_only),
            "y": _dump_embeddings(y0, shape_only),
            "c": _dump_embeddings(c0, shape_only),
            "valid_cluster_ids": list(self.__clusters.keys()),
            "clusters": [c.dump(shape_only) for c in self.__clusters.values()],
            "persons": [p.dump(shape_only) for p in self.__persons.values()],
            "unprocessed_embeddings": _dump_embeddings(self.__unprocessed_embeddings, shape_only),
            "unassigned_embeddings": _dump_embeddings(self.__unassigned_embeddings, shape_only)
        }

    def dump_state(self, shape_only=False):
        with self.__lock0:
            return self.__dump_state(shape_only)

    def plot(self):
        with self.__lock0:
            x0, cluster_emb_len = self.__get_all_embeddings_for_mapping()
            d = self.__dump_state(shape_only=False)

        tsne = TSNE(n_components=2, metric="cosine")
        x0m = tsne.fit_transform(x0)
        x0h = _embedding_hash(x0)

        pl = _Plot(x0m, x0h)
        return pl.plot(d=d)


class SpeechbrainEmbeddingDatabase(EmbeddingDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(192, *args, **kwargs)


class PyannoteEmbeddingDatabase(EmbeddingDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(512, *args, **kwargs)


class HybridEmbeddingDatabase:
    def __init__(self, database_dir_name: str,
                 param_for_speechbrain: dict | None = None, param_for_pyannote: dict | None = None):
        self.__db: list[EmbeddingDatabase] = [
            SpeechbrainEmbeddingDatabase(
                database_file_name=os.path.join(database_dir_name, "embedding_speechbrain.pickle"),
                **param_for_speechbrain),
            PyannoteEmbeddingDatabase(
                database_file_name=os.path.join(database_dir_name, "embedding_pyannote.pickle"),
                **param_for_pyannote),
        ]

    @staticmethod
    def __db_index(embedding: np.ndarray):
        if len(embedding) == 192:
            return 0
        if len(embedding) == 512:
            return 1
        raise ValueError()

    @staticmethod
    def __db_index_from_name(embedding_type: str):
        if embedding_type == "speechbrain":
            return 0
        if embedding_type == "pyannote":
            return 1
        raise ValueError()

    @staticmethod
    def __encode_person_id(index: int, person_id: int):
        assert person_id != -1
        return (index + 1) * 1000000 + person_id

    def __decode_index(self, converted_person_id: int):
        index = converted_person_id // 1000000 - 1
        if index < 0 or len(self.__db) <= index:
            raise ValueError()
        return index

    @staticmethod
    def __decode_person_id(converted_person_id: int):
        return converted_person_id % 1000000

    def metrics(self, emb0, emb1):
        index0 = self.__db_index(emb0)
        index1 = self.__db_index(emb1)
        return self.__db[index0].metrics(emb0, emb1) if index0 == index1 else 2.0

    def map(self, embeddings: list[np.ndarray] | np.ndarray, update=True) -> list[[int, str | None]]:
        emb_list = [[] for _ in range(len(self.__db))]
        emb_index = [[] for _ in range(len(self.__db))]
        for i in range(len(embeddings)):
            index = self.__db_index(embeddings[i])
            emb_list[index].append(embeddings[i])
            emb_index[index].append(i)

        ret = [None] * len(embeddings)
        for index in range(len(self.__db)):
            if len(emb_list[index]) == 0:
                continue
            for j, e in enumerate(self.__db[index].map(emb_list[index], update=update)):
                ret[emb_index[index][j]] = [-1 if e[0] == -1 else self.__encode_person_id(index, e[0]), e[1]]
        return ret

    def get_persons(self) -> list[Person]:
        ret = []
        for index, db in enumerate(self.__db):
            for p in db.get_persons():
                p.person_id = self.__encode_person_id(index, p.person_id)
                p.superseded_by = -1 if p.superseded_by == -1 else self.__encode_person_id(index, p.superseded_by)
                ret.append(p)
        return ret

    def rename(self, person_id: int, new_name: str):
        self.__db[self.__decode_index(person_id)].rename(self.__decode_person_id(person_id), new_name)

    def erase(self, person_id: int):
        self.__db[self.__decode_index(person_id)].erase(self.__decode_person_id(person_id))

    def sync(self):
        for db in self.__db:
            db.sync()

    def reconstruct(self, **kwargs):
        for db in self.__db:
            db.reconstruct(**kwargs)

    def get_generation(self):
        return [db.get_generation() for db in self.__db]

    def dump_state(self):
        return [db.dump_state for db in self.__db]

    def plot(self, embedding_type: str):
        return self.__db[self.__db_index_from_name(embedding_type)].plot()


def _create_embedding_map_from_pickle(d):
    embeddings0 = []
    if len(d["unprocessed_embeddings"]) != 0:
        embeddings0.append(np.vstack(d["unprocessed_embeddings"]))
    if d["unassigned_embeddings"] is not None:
        embeddings0.append(d["unassigned_embeddings"])
    for c in d["clusters"].values():
        embeddings0.append(c.core_embeddings)
        if c.sub_embeddings is not None:
            embeddings0.append(c.sub_embeddings)

    embeddings1 = []
    for p in d["persons"].values():
        embeddings1.append(p.core_embeddings)

    x0 = np.vstack(embeddings0)
    cluster_emb_len = len(x0)
    x0i = [i for i in range(cluster_emb_len)]
    random.shuffle(x0i)
    x0 = x0[x0i]

    x0 = np.concatenate([x0, np.vstack(embeddings1)], axis=0)
    logging.info("found %d embeddings (%d in clusters), shape = %d" % (x0.shape[0], cluster_emb_len, x0.shape[1]))

    # tsne = TSNE(n_components=2, metric="precomputed", init="random")
    # x0m = tsne.fit_transform(_cosine_distance_mat(x0))
    tsne = TSNE(n_components=2, metric="cosine")
    x0m = tsne.fit_transform(x0)
    x0h = _embedding_hash(x0)
    return x0, x0m, x0h, cluster_emb_len


def _create_embedding_database(db_opt: dict, embeddings_ref_for_shape: np.ndarray):
    length = embeddings_ref_for_shape.shape[1]
    if length == 192:
        return SpeechbrainEmbeddingDatabase(**db_opt)
    if length == 512:
        return PyannoteEmbeddingDatabase(**db_opt)
    raise ValueError()


def _find_embedding_length(d):
    if d["unassigned_embeddings"] is not None:
        return d["unassigned_embeddings"].shape[1]
    if len(d["unprocessed_embeddings"]) != 0:
        return len(d["unprocessed_embeddings"][0])
    for c in d["clusters"].values():
        return c.core_embeddings.shape[1]


def _load_embedding_database(input_file_name):
    with open(input_file_name, "rb") as f:
        d = pickle.load(f)
    length = _find_embedding_length(d)
    if length == 192:
        return SpeechbrainEmbeddingDatabase(database_file_name=input_file_name)
    if length == 512:
        return PyannoteEmbeddingDatabase(database_file_name=input_file_name)


class _Plot:
    def __init__(self, x0m, x0h):
        self.__x0m = x0m
        self.__x0h = x0h

        fig, ax = plt.subplots(dpi=300)
        self.__fig = fig
        self.__ax = ax

        plt.subplots_adjust(left=0.05, right=0.975, bottom=0.05, top=0.975)
        plt.xlim(x0m[:, 0].min() - 1, x0m[:, 0].max() + 1)
        plt.ylim(x0m[:, 1].min() - 1, x0m[:, 1].max() + 1)
        plt.xticks(size=6.0)
        plt.yticks(size=6.0)

        self.__color_hue_step = 43
        self.__color_index = random.randint(0, self.__color_hue_step - 1)
        self.__cluster_color_map = {}
        self.__artists = []

    def plot(self, db: EmbeddingDatabase | None = None, d=None):
        if db is not None:
            d = db.dump_state()
        x1 = d["x"]
        y1 = d["y"]
        c1 = d["c"]

        x1h = _embedding_hash(x1)
        _, indexes0, indexes1 = np.intersect1d(self.__x0h, x1h, return_indices=True)
        if len(indexes1) != len(x1h):
            logging.warning("hash collision (or duplicated vectors) in %d embeddings" % (len(x1h) - len(indexes1)))
        # x1 = x1[indexes1]
        y1 = y1[indexes1]
        c1 = c1[indexes1]
        x1m = self.__x0m[indexes0]

        elements = []

        def _add_scatter_c(mask_, color_, s_, **kwargs_):
            nonlocal elements
            tx_ = x1m[mask_]
            if len(tx_) != 0:
                elements.append(self.__ax.scatter(tx_[:, 0], tx_[:, 1], color=color_, s=s_, **kwargs_))

        _add_scatter_c(y1 == -1, "lightgray", 2.0)
        _add_scatter_c(y1 == -2, "black", 2.0)

        for cluster_id in sorted(d["valid_cluster_ids"]):
            if cluster_id not in self.__cluster_color_map:
                cr, cg, cb = colorsys.hsv_to_rgb(self.__color_index / 360, 0.75, 0.75)
                color0 = "#%02x%02x%02x" % (int(cr * 255), int(cg * 255), int(cb * 255))
                cr, cg, cb = colorsys.hsv_to_rgb(self.__color_index / 360, 1.00, 0.50)
                color1 = "#%02x%02x%02x" % (int(cr * 255), int(cg * 255), int(cb * 255))
                self.__cluster_color_map[cluster_id] = [color0, color1]
                self.__color_index += self.__color_hue_step
            else:
                color0, color1 = self.__cluster_color_map[cluster_id]

            _add_scatter_c((y1 == cluster_id) * np.logical_not(c1), color0, 3.0)
            _add_scatter_c((y1 == cluster_id) * c1, color0, 5.0, edgecolors=color1, linewidths=0.5)

        sorted_persons = sorted(d["persons"], key=lambda p_: p_["person_id"])
        x2 = np.vstack([p["core_embeddings"] for p in sorted_persons])
        y2 = np.concatenate([
            np.full((len(p["core_embeddings"]),), p["person_id"], dtype=np.int32) for p in sorted_persons])
        x2h = _embedding_hash(x2)
        _, indexes0, indexes1 = np.intersect1d(self.__x0h, x2h, return_indices=True)
        # x2 = x2[indexes1]
        y2 = y2[indexes1]
        x2m = self.__x0m[indexes0]

        def _add_scatter_p(mask_, color_, **kwargs_):
            nonlocal elements
            tx_ = x2m[mask_]
            if len(tx_) != 0:
                elements.append(self.__ax.scatter(
                    tx_[:, 0], tx_[:, 1], color=color_, edgecolors="#C0C0C0", s=7.0, linewidths=0.25, **kwargs_))

        def _add_scatter_p_no_cluster(mask_, **kwargs_):
            nonlocal elements
            tx_ = x2m[mask_]
            if len(tx_) != 0:
                elements.append(self.__ax.scatter(
                    tx_[:, 0], tx_[:, 1], marker="o", facecolors="none", edgecolors="red", s=7.0, linewidths=0.25,
                    **kwargs_))

        def _add_label(mask_, color_, text_, **kwargs):
            nonlocal elements
            tx_ = x2m[mask_]
            if len(tx_) != 0:
                tx_ = np.average(tx_, axis=0)
                elements.append(self.__ax.text(tx_[0], tx_[1], text_, color=color_, bbox={
                    "facecolor": "white", "edgecolor": "none", "boxstyle": "Round", "alpha": 0.5}, **kwargs))

        for p in sorted_persons:
            if p["cluster_id"] != -1:
                _add_scatter_p(y2 == p["person_id"], self.__cluster_color_map[p["cluster_id"]][1])
            else:
                _add_scatter_p_no_cluster(y2 == p["person_id"])

        for p in sorted_persons:
            _add_label(
                y2 == p["person_id"],
                "gray" if p["superseded_by"] != -1 else "black" if p["cluster_id"] != -1 else "red",
                "%s[%d]" % (p["name"], p["person_id"]), size=6.0)

        self.__artists.append(elements)
        return self.__fig

    def plot_embeddings(self, embeddings, **kwargs):
        x1h = _embedding_hash(embeddings)
        _, indexes0, _ = np.intersect1d(self.__x0h, x1h, return_indices=True)
        x1m = self.__x0m[indexes0]
        self.__artists.append([self.__ax.scatter(x1m[:, 0], x1m[:, 1], **kwargs)])
        return self.__fig

    def save(self, file_name):
        if len(self.__artists) <= 1:
            plt.savefig(file_name + ".png", dpi=300)
        else:
            anim = ArtistAnimation(self.__fig, self.__artists, interval=200)
            anim.save(file_name + ".mp4")

    @staticmethod
    def show():
        plt.show()


def _op_dump(db):
    print(json.dumps(db.dump_state(shape_only=True), indent=4, ensure_ascii=False))


def _op_reconstruct_common(input_file_name, db_opt, inherit_persons, manipulator):
    with open(input_file_name, "rb") as f:
        d = pickle.load(f)

    x0, x0m, x0h, cluster_emb_len = _create_embedding_map_from_pickle(d)

    db = _create_embedding_database(db_opt, x0)
    if inherit_persons:
        for p in d["persons"].values():
            if p.name != p.default_name:
                db.add_person(p.core_embeddings, p.name)

    if manipulator is not None:
        manipulator(db)

    return x0, x0m, x0h, cluster_emb_len, db


def _op_reconstruct_and_plot(
        input_file_name, db_opt, manipulator, inherit_persons, save_plot_file_name, show, force_reduce):

    x0, x0m, x0h, cluster_emb_len, db = _op_reconstruct_common(input_file_name, db_opt, inherit_persons, manipulator)

    db.map(x0[:cluster_emb_len])
    removed = db.reconstruct(force_reconstruct=True, force_reduce=force_reduce)

    if save_plot_file_name is not None or show:
        pl = _Plot(x0m, x0h)
        if removed is not None:
            pl.plot_embeddings(removed, color="gray", marker="x", s=2.0, linewidths=0.2)
        pl.plot(db=db)
        if save_plot_file_name is not None:
            pl.save(save_plot_file_name)
        if show:
            pl.show()

    return db


def _op_reconstruct_and_plot_anim(
        input_file_name, db_opt, manipulator, inherit_persons, sample_pitch, save_plot_file_name, show):

    x0, x0m, x0h, cluster_emb_len, db = _op_reconstruct_common(input_file_name, db_opt, inherit_persons, manipulator)

    pl = _Plot(x0m, x0h)
    for i in range(0, cluster_emb_len, sample_pitch):
        i_bound = min(cluster_emb_len, i + sample_pitch)
        db.map(x0[i:i_bound])
        db.reconstruct(force_reconstruct=True)
        pl.plot(db=db)

    if save_plot_file_name is not None:
        pl.save(save_plot_file_name)
    if show:
        pl.show()

    return db


def main(args=None):
    parser = argparse.ArgumentParser(description="Tool for embeddings database")
    parser.add_argument(
        "--input", metavar="FILE_NAME", action="store",
        dest="input", required=True)
    parser.add_argument(
        "--output", metavar="FILE_NAME", action="store",
        dest="output", default=None)
    parser.add_argument("--overwrite", action="store_true", dest="overwrite")
    parser.add_argument("--show", action="store_true", dest="show")
    parser.add_argument(
        "--save-plot", metavar="FILE_NAME", action="store",
        dest="save_plot", default=None)

    gr_op = parser.add_argument_group("Inspect or modify database options")
    gr_op.add_argument(
        "-D", "--dump", action="store_true",
        dest="op_dump")
    gr_op.add_argument(
        "-R", "--reconstruct", action="store_true",
        dest="op_reconstruct")
    gr_op.add_argument(
        "--force-reduce", action="store_true",
        dest="op_force_reduce")
    gr_op.add_argument(
        "--reconstruct-anim", metavar="reconstruct_per_sample_count[1..]",
        action="store", type=int,
        dest="op_reconstruct_anim", default=-1)
    gr_op.add_argument(
        "--erase-person-id", metavar="ID or \"all\"", action="append", type=str,
        dest="op_erase_person_id", default=[])

    gr_db = parser.add_argument_group("Database parameters")
    gr_db.add_argument(
        "--threshold", metavar="float[0..1]", action="store", type=float,
        dest="emb_db__threshold", default=0.6)
    gr_db.add_argument(
        "--dbscan-eps", metavar="float[0..1]", action="store", type=float,
        dest="emb_db__dbscan_eps", default=0.4)
    gr_db.add_argument(
        "--dbscan-min-samples", metavar="int[2..]", action="store", type=int,
        dest="emb_db__dbscan_min_samples", default=6)
    gr_db.add_argument(
        "--min-matched-embeddings-to-inherit-cluster", metavar="int[2..]", action="store", type=int,
        dest="emb_db__min_matched_embeddings_to_inherit_cluster", default=3)
    gr_db.add_argument(
        "--min-matched-embeddings-to-match-person", metavar="int[2..]", action="store", type=int,
        dest="emb_db__min_matched_embeddings_to_match_person", default=3)

    gr_ext = parser.add_argument_group("Extra options")
    gr_ext.add_argument("--inherit-persons", action="store_true", dest="ext_inherit_persons")

    opt = parser.parse_args(args=args)
    db_opt = {k[8:]: v for k, v in vars(opt).items() if k.startswith("emb_db__")}
    db = None

    if opt.op_dump:
        _op_dump(_load_embedding_database(opt.input))
        return

    def _manipulation(db_):
        for person_id in opt.op_erase_person_id:
            if person_id == "all":
                db_.erase_all()
            else:
                db_.erase(int(person_id))

    if opt.op_reconstruct:
        db = _op_reconstruct_and_plot(
            opt.input, db_opt, _manipulation, opt.ext_inherit_persons,
            opt.save_plot, opt.show, opt.op_force_reduce)

    if opt.op_reconstruct_anim > 0:
        db = _op_reconstruct_and_plot_anim(
            opt.input, db_opt, _manipulation, opt.ext_inherit_persons,
            opt.op_reconstruct_anim, opt.save_plot, opt.show)

    if db is None:
        raise ValueError()

    if opt.overwrite:
        os.rename(opt.input, opt.input + datetime.datetime.fromtimestamp(time.time()).strftime(".backup.%Y%m%d.%H%M"))
        db.sync(opt.input)
        return

    if opt.output is not None:
        db.sync(opt.output)


def plot(db):
    if not isinstance(db, EmbeddingDatabase):
        raise ValueError()
