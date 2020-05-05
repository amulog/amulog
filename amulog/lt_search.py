
from collections import deque
import logging

from amulog import lt_common

_logger = logging.getLogger(__package__)


class LTSearch(object):

    def __init__(self):
        self._l_lt = []

    def add(self, ltid, l_w):
        self._l_lt.append((ltid, l_w))

    def search(self, l_w):
        for ltid, tmp_l_w in self._l_lt:
            if len(l_w) == len(tmp_l_w):
                for w1, w2 in zip(l_w, tmp_l_w):
                    if w1 == w2:
                        pass
                    elif w2 == lt_common.REPLACER:
                        pass
                    else:
                        break
                else:
                    return ltid
        else:
            return None

    def shuffle(self):
        import random
        random.shuffle(self._l_lt)


class LTSearchTree(LTSearch):
    # Search tree for un-incremental lt generation algorithms

    def __init__(self):
        self.root = self._new_node()

    def __str__(self):
        l_buf = []

        def print_children(point, depth, l_sparent):
            word = point.word
            if word is None:
                word = lt_common.REPLACER

            cnt = point.child_num()
            if cnt == 1:
                l_sparent.append(word)
            else:
                l_sparent.append(word)
                buf = "-" * (depth - len(l_sparent) + 1) + \
                      " {0}".format(" ".join(l_sparent))
                if point.end is not None:
                    buf += "  <-- ltid {0}".format(point.end)
                l_buf.append(buf)
                l_sparent = []

            for word in point.windex.keys():
                print_children(point.windex[word], depth + 1, l_sparent[:])
            if point.wild is not None:
                print_children(point.wild, depth + 1, l_sparent[:])

        point = self.root
        l_buf.append("<head of log template search tree>")
        for w in point.windex.keys():
            print_children(point.windex[w], 0, [])
        if point.wild is not None:
            print_children(point.wild, 0, [])
        return "\n".join(l_buf)

    @staticmethod
    def _new_node(parent=None, word=None):
        return LTSearchTreeNode(parent, word)

    def add(self, ltid, l_w):
        point = self.root
        for w in l_w:
            if w == lt_common.REPLACER:
                if point.wild is None:
                    point.wild = self._new_node(point, w)
                point = point.wild
            else:
                if not w in point.windex:
                    point.windex[w] = self._new_node(point, w)
                point = point.windex[w]
        else:
            point.set_ltid(ltid)

    def _trace(self, ltwords):
        _logger.debug("tracing message {0}".format(ltwords))
        point = self.root
        tmp_ltwords = deque(ltwords)
        check_points = []
        # points with 2 candidates to go, for example "word" and "**"
        # [(node, len(left_words)), ...]
        # use as stack
        while True:
            w = tmp_ltwords.popleft()
            if w == lt_common.REPLACER:
                # w is sym (This means the input is not a message but a template)
                # go wild node
                if point.wild is None:
                    return None
                else:
                    point = point.wild
            elif w in point.windex:
                # w is in windex
                # go the word node
                # also go to wild node, when check_points poped
                if point.wild is not None:
                    check_points.append((point, deque(tmp_ltwords)))
                    _logger.debug("# add checkpoint ({0} (+{1}))".format(point, len(tmp_ltwords) + 1))
                _logger.debug("{0} (+{1}) -> goto {2}".format(
                    point, len(tmp_ltwords) + 1, w))
                point = point.windex[w]
            elif point.wild is not None:
                # w is not in windex, but have wild node
                _logger.debug("{0} (+{1}) -> goto {2}".format(
                    point, len(tmp_ltwords) + 1, lt_common.REPLACER))
                point = point.wild
            else:
                # no template to process w, go back or end
                _logger.debug("{0} (+{1}) : no available children".format(point, len(tmp_ltwords) + 1))
                if len(check_points) == 0:
                    _logger.debug("template not found")
                    return None
                else:
                    p, tmp_ltwords = check_points.pop()
                    # tmp_ltwords = deque(ltwords[-left_wlen:]) if left_wlen > 0 else []
                    #    # +1 : for one **(wild) node
                    point = p.wild
                    _logger.debug("go back to a stacked branch : {0} (+{1})".format(point, len(tmp_ltwords)))
                    _logger.debug("remaining words : {0}".format(tmp_ltwords))

            while len(tmp_ltwords) == 0:
                if point.end is None:
                    _logger.debug("{0} (+{1}) : no template in this node".format(point, len(tmp_ltwords)))
                    if len(check_points) == 0:
                        _logger.debug("all ends are empty(no templates)")
                        return None
                    else:
                        p, tmp_ltwords = check_points.pop()
                        # tmp_ltwords = deque(ltwords[-left_wlen:]) if left_wlen > 0 else []
                        point = p.wild
                        _logger.debug("go back to a stacked branch : {0} (+{1})".format(point, len(tmp_ltwords)))
                        _logger.debug("remaining words : {0}".format(tmp_ltwords))
                else:
                    _logger.debug("done (tpl: {0}, id: {1})".format(point, point.end))
                    return point

    def remove(self, ltid, l_w):
        point = self._trace(l_w)
        if point is None:
            _logger.warning(
                "LTSearchTree : Failed to remove ltid {0}".format(ltid))
        point.remove_ltid(ltid)
        while point.unnecessary():
            w = point.word
            point = point.parent
            if w is None:
                point.wild = None
            else:
                point.wdict.pop(w)
        else:
            if self.root is None:
                self.root = self._new_node()

    def search(self, l_w):
        node = self._trace(l_w)
        if node is None:
            return None
        else:
            return node.get_ltid()

    def shuffle(self):
        pass


class LTSearchTreeNode:

    def __init__(self, parent, word):
        self.windex = {}
        self.wild = None
        self.end = None
        self.parent = parent  # for reverse search to remove
        self.word = word

    def __str__(self):
        ret = []
        p = self
        while p.parent is not None:
            if p.word is None:
                ret.append("<root>")
            else:
                ret.append(p.word)
            p = p.parent
        return " ".join(reversed(ret))

    def child(self, word=None):
        if word is None:
            # wildcard
            return self.wild
        elif word in self.windex:
            return self.windex[word]
        else:
            return None

    def child_num(self):
        cnt = len(self.windex.keys())
        if self.wild is not None:
            cnt += 1
        return cnt

    def current_point(self):
        buf = []
        point = self
        while point.parent is not None:
            buf = [point.word] + buf
            point = point.parent
        print(" ".join(buf))

    def set_ltid(self, ltid):
        self.end = ltid

    def remove_ltid(self, ltid):
        assert self.end == ltid
        self.end = None

    def get_ltid(self):
        return self.end

    def unnecessary(self):
        return (len(self.windex) == 0) and \
               (self.wild is None) and \
               (self.end is None)


def init_searcher(name):
    if name == "tree":
        return LTSearchTree()
    elif name == "table":
        return LTSearch()
    else:
        raise NotImplementedError
