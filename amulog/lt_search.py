
from collections import deque
import logging

from amulog import lt_common

_logger = logging.getLogger(__package__)


class LTSearch(object):

    def __init__(self):
        self._init_ltsearch()

    def _init_ltsearch(self):
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

    def _init_ltsearch(self):
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
                if w not in point.windex:
                    point.windex[w] = self._new_node(point, w)
                point = point.windex[w]
        else:
            point.set_ltid(ltid)

    def _trace(self, ltwords):
        # _logger.debug("LTSearchTree: trace {0}".format(ltwords))
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
                    # msg = "add checkpoint ({0} (+{1}))".format(
                    #     point, len(tmp_ltwords) + 1)
                    # _logger.debug("LTSearchTree: " + msg)
                    # msg = "{0} (+{1}) -> goto {2}".format(
                    #     point, len(tmp_ltwords) + 1, w)
                    # _logger.debug("LTSearchTree: " + msg)
                point = point.windex[w]
            elif point.wild is not None:
                # w is not in windex, but have wild node
                # msg = "{0} (+{1}) -> goto {2}".format(
                #     point, len(tmp_ltwords) + 1, lt_common.REPLACER)
                # _logger.debug("LTSearchTree: " + msg)
                point = point.wild
            else:
                # no template to process w, go back or end
                # msg = "{0} (+{1}) : no available children".format(
                #     point, len(tmp_ltwords) + 1)
                # _logger.debug("LTSearchTree: " + msg)
                if len(check_points) == 0:
                    # _logger.debug("LTSearchTree: template not found")
                    return None
                else:
                    p, tmp_ltwords = check_points.pop()
                    # tmp_ltwords = deque(ltwords[-left_wlen:]) if left_wlen > 0 else []
                    #    # +1 : for one **(wild) node
                    point = p.wild
                    # msg = "go back to a stacked branch : {0} (+{1})".format(
                    #     point, len(tmp_ltwords))
                    # _logger.debug("LTSearchTree: " + msg)
                    # msg = "remaining words : {0}".format(tmp_ltwords)
                    # _logger.debug("LTSearchTree: " + msg)

            while len(tmp_ltwords) == 0:
                if point.end is None:
                    # msg = "LTSearchTree: {0} (+{1}) : no template in this node".format(
                    #     point, len(tmp_ltwords))
                    # _logger.debug("LTSearchTree: " + msg)
                    if len(check_points) == 0:
                        # msg = "all ends are empty(no templates)"
                        # _logger.debug("LTSearchTree: " + msg)
                        return None
                    else:
                        p, tmp_ltwords = check_points.pop()
                        # tmp_ltwords = deque(ltwords[-left_wlen:]) if left_wlen > 0 else []
                        point = p.wild
                        # msg = "go back to a stacked branch : {0} (+{1})".format(
                        #     point, len(tmp_ltwords))
                        # _logger.debug("LTSearchTree: " + msg)
                        # msg = "remaining words : {0}".format(tmp_ltwords)
                        # _logger.debug("LTSearchTree: " + msg)
                else:
                    # msg = "done (tpl: {0}, id: {1})".format(point, point.end)
                    # _logger.debug("LTSearchTree: " + msg)
                    return point

    def remove(self, ltid, l_w):
        point = self._trace(l_w)
        if point is None:
            _logger.warning(
                "LTSearchTree: Failed to remove ltid {0}".format(ltid))
        point.remove_ltid(ltid)
        while point.unnecessary():
            w = point.word
            point = point.parent
            if w is None:
                point.wild = None
            else:
                point.windex.pop(w)
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


class LTSearchTreeNew(LTSearch):

    KEY_WILDCARD = "@wild"
    KEY_TEMPLATE_ID = "@ltid"

    def _init_ltsearch(self):
        self.root = {}

    def add(self, ltid, l_w):
        current = self.root
        for w in l_w:
            if w == lt_common.REPLACER:
                key = self.KEY_WILDCARD
            else:
                key = w
            if key not in current:
                current[key] = {}
            current = current[key]
        else:
            current[self.KEY_TEMPLATE_ID] = ltid

    def _trace(self, ltwords):
        current = self.root
        tmp_ltwords = deque(ltwords)
        stack_path = []
        # points with 2 candidates to go, for example "word" and "**"
        # [(node, len(left_words)), ...]

        while True:
            w = tmp_ltwords.popleft()
            if w == lt_common.REPLACER:
                # w is wildcard (This means the input is not a message but a template)
                # go wildcard node
                if self.KEY_WILDCARD in current:
                    current = current[self.KEY_WILDCARD]
                else:
                    return None
            elif w in current:
                # w is in children of current node
                # go the word or wildcard node
                # If both exists, push wildcard to stack
                if self.KEY_WILDCARD in current:
                    stack_path.append((current, deque(tmp_ltwords)))
                current = current[w]
            elif self.KEY_WILDCARD in current:
                # w is not in children, but have wildcard node
                # go wildcard node
                current = current[self.KEY_WILDCARD]
            else:
                # no template to match, go back with stack or end
                if not stack_path:
                    return None
                else:
                    node, tmp_ltwords = stack_path.pop()
                    current = node[self.KEY_WILDCARD]

            while not tmp_ltwords:
                # not "if" but "while"
                # because current may be end node just after stack-pop
                # it happens when last word matches both a word and wildcard
                if self.KEY_TEMPLATE_ID in current:
                    # template match successfully
                    return current[self.KEY_TEMPLATE_ID]
                elif not stack_path:
                    return None
                else:
                    # no template to match, go back with stack or end
                    node, tmp_ltwords = stack_path.pop()
                    current = node[self.KEY_WILDCARD]

    def _trace_path(self, ltwords):
        current = self.root
        tmp_ltwords = deque(ltwords)
        stack_path = []
        ret_path = []
        # points with 2 candidates to go, for example "word" and "**"
        # [(node, len(left_words), ret_path), ...]

        while True:
            w = tmp_ltwords.popleft()
            if w == lt_common.REPLACER:
                # w is wildcard (This means the input is not a message but a template)
                # go wildcard node
                if self.KEY_WILDCARD in current:
                    current = current[self.KEY_WILDCARD]
                else:
                    return None
            elif w in current:
                # w is in children of current node
                # go the word or wildcard node
                # If both exists, push wildcard to stack
                if self.KEY_WILDCARD in current:
                    stack_path.append((current, deque(tmp_ltwords), ret_path))
                current = current[w]
            elif self.KEY_WILDCARD in current:
                # w is not in children, but have wildcard node
                # go wildcard node
                current = current[self.KEY_WILDCARD]
            else:
                # no template to match, go back with stack or end
                if not stack_path:
                    return None
                else:
                    node, tmp_ltwords, ret_path = stack_path.pop()
                    current = current[self.KEY_WILDCARD]

            ret_path.append(current)

            while not tmp_ltwords:
                if self.KEY_TEMPLATE_ID in current:
                    # template match successfully
                    return ret_path
                elif not stack_path:
                    return None
                else:
                    # no template to match, go back with stack or end
                    node, tmp_ltwords, ret_path = stack_path.pop()
                    current = current[self.KEY_WILDCARD]
                    ret_path.append(current)

    def search(self, words):
        return self._trace(words)

    def remove(self, tpl):
        path = self._trace_path(tpl)
        if path is None:
            _logger.warning(
                "LTSearchTree: Failed to remove tpl {0}".format(tpl))
            return

        ltid = path[-1].pop(self.KEY_TEMPLATE_ID)
        rev_tpl = list(reversed(tpl))
        rev_path = list(reversed(path))
        rev_path_parent = rev_path[1:] + [self.root]
        for w, node, parent_node in zip(rev_tpl, rev_path, rev_path_parent):
            if w == lt_common.REPLACER:
                w = self.KEY_WILDCARD
            if not node:
                # node empty: remove link from parent node
                parent_node.pop(w)
        return ltid

    def shuffle(self):
        pass


def init_searcher(name):
    if name == "tree":
        return LTSearchTreeNew()
    elif name == "table":
        return LTSearch()
    else:
        raise NotImplementedError
