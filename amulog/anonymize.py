import json


class AnonymizeMapper:

    def __init__(self, conf, online_batchsize=1000):
        self._conf = conf
        self._online_batchsize = online_batchsize
        self._d_host = None
        self._d_lt = None

        self._overwrite_method = conf["visual"]["anonymize_overwrite_method"]

    def _filepath(self):
        return self._conf["visual"]["anonymize_mapping_file"]

    def load(self):
        fp = self._filepath()
        with open(fp, mode='r', encoding='utf-8') as f:
            obj = json.load(f)
            self._d_host = obj["host_mapping"]
            self._d_lt = obj["lt_mapping"]

    def dump(self):
        fp = self._filepath()
        with open(fp, mode='wt', encoding='utf-8') as f:
            obj = {"host_mapping": self._d_host,
                   "lt_mapping": self._d_lt}
            json.dump(obj, f, indent=2)
        return fp

    def _generate_mapping(self, src_ld):
        host_mapping = {}
        host_mapping_dump = {}
        hosts = sorted(src_ld.whole_host())
        for num, host in enumerate(hosts):
            replaced_host = "host" + str(num).zfill(len(str(len(hosts))))
            host_mapping[host] = replaced_host
            host_mapping_dump[replaced_host] = host

        lt_mapping = {}
        lt_mapping_dump = {}

        from . import lt_common
        for ltobj in src_ld.iter_lt():
            new_ltw = [lt_common.REPLACER if w == lt_common.REPLACER
                       else lt_common.ANONYMIZED_DESC
                       for w in ltobj.ltw]
            new_lts = [" "] * len(ltobj.lts)
            new_lts[0] = ""
            new_lts[-1] = ""
            lt_mapping[ltobj.ltid] = {"ltw": new_ltw,
                                      "lts": new_lts}
            lt_mapping_dump[str(ltobj.ltid)] = (ltobj.ltw, ltobj.lts)

        self._host_mapping = host_mapping
        self._lt_mapping = lt_mapping
        self._d_host = host_mapping_dump
        self._d_lt = lt_mapping_dump

    def _anonymize_overwrite_legacy(self, ld):
        # overwrite hosts
        hosts = set(ld.whole_host())
        for num, host in enumerate(hosts):
            replaced_host = self._host_mapping[host]
            ld.db.update_log({"host": host}, host=replaced_host)
            ld.commit_db()

        # overwrite lt and log words
        from . import lt_common
        for ltobj in ld.iter_lt():
            new_ltw = self._lt_mapping[ltobj.ltid]["ltw"],
            new_lts = self._lt_mapping[ltobj.ltid]["lts"],
            new_ltobj = lt_common.LogTemplate(ltobj.ltid, ltobj.ltgid,
                                              new_ltw, new_lts, ltobj.count)
            ld.lttable.update_lt(new_ltobj)
            ld.db.update_lt(ltid=ltobj.ltid, ltw=new_ltw, lts=new_lts,
                            count=ltobj.count)

            ld.db.update_log({"ltid": ltobj.ltid}, l_w=new_ltw)
            ld.commit_db()

    def _anonymize_overwrite(self, ld):
        online_batchsize = ld.conf.getint("manager",
                                          "online_batchsize")

        ld.db.switch_temporal_table(ld.db.tablename_lt)
        from . import lt_common
        for ltobj in ld.iter_lt():
            new_ltobj = lt_common.LogTemplate(
                ltid=ltobj.ltid,
                ltgid=ltobj.ltgid,
                ltw=self._lt_mapping[ltobj.ltid]["ltw"],
                lts=self._lt_mapping[ltobj.ltid]["lts"],
                count=ltobj.count
            )
            ld.db.add_lt(new_ltobj)  # only update lt (w/o ltg)
            ld.commit_db()
        ld.db.apply_temporal_table(ld.db.tablename_lt)

        ld.db.switch_temporal_table(ld.db.tablename_log)
        online_counter = 0
        for lm in ld.iter_all():
            ld.add_line(lid=lm.lid,
                        ltid=lm.lt.ltid,
                        dt=lm.dt,
                        host=self._host_mapping[lm.host],
                        l_w=self._lt_mapping[lm.lt.ltid]["ltw"])
            online_counter += 1
            if online_counter >= online_batchsize:
                ld.commit_db()
                online_counter = 0
        ld.db.apply_temporal_table(ld.db.tablename_log)

    def _anonymize_migration(self, ld_src, ld_dst):
        online_batchsize = ld_dst.conf.getint("manager", "online_batchsize")

        from . import lt_common
        for ltobj in ld_src.iter_lt():
            new_ltobj = lt_common.LogTemplate(
                ltid=ltobj.ltid,
                ltgid=ltobj.ltgid,
                ltw=self._lt_mapping[ltobj.ltid]["ltw"],
                lts=self._lt_mapping[ltobj.ltid]["lts"],
                count=ltobj.count
            )
            ld_dst.db.add_lt(new_ltobj)  # only update lt (w/o ltg)
            ld_dst.commit_db()

        for ltid, ltgid in ld_src.db.iter_ltg_def():
            ld_dst.db.add_ltg(ltid, ltgid)
            ld_dst.commit_db()

        for ltid, tag in ld_src.db.iter_tag_def():
            ld_dst.db.add_tags(ltid, [tag])
            ld_dst.commit_db()

        online_counter = 0
        for lm in ld_src.iter_all():
            ld_dst.add_line(lid=lm.lid,
                            ltid=lm.lt.ltid,
                            dt=lm.dt,
                            host=self._host_mapping[lm.host],
                            l_w=self._lt_mapping[lm.lt.ltid]["ltw"])
            online_counter += 1
            if online_counter >= online_batchsize:
                ld_dst.commit_db()
                online_counter = 0

    def anonymize(self, conf_dst=None):
        from . import log_db
        if conf_dst:
            ld = log_db.LogData(self._conf, edit=False)
            ld_dst = log_db.LogData(conf_dst, edit=True, reset_db=True)
            self._generate_mapping(ld)
            self._anonymize_migration(ld, ld_dst)
        else:
            ld = log_db.LogData(self._conf, edit=True)
            self._generate_mapping(ld)
            if self._overwrite_method == "standard":
                self._anonymize_overwrite(ld)
            elif self._overwrite_method == "legacy":
                self._anonymize_overwrite_legacy(ld)
            else:
                raise ValueError

    def mapping(self):
        from . import log_db
        ld = log_db.LogData(self._conf, edit=False)
        self._generate_mapping(ld)

    def restore_host(self, host):
        return self._d_host[host]

    def restore_lt(self, ltobj):
        from . import lt_common
        return lt_common.LogTemplate(
            ltid=ltobj.ltid,
            ltgid=ltobj.ltgid,
            ltw=self._d_lt[str(ltobj.ltid)][0],
            lts=self._d_lt[str(ltobj.ltid)][1],
            count=ltobj.count
        )
