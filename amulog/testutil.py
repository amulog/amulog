import os
import datetime
import random
import re
import configparser
import numpy as np

from amulog import config

DEFAULT_CONFIG = "/".join((os.path.dirname(os.path.abspath(__file__)),
                           "./data/testlog.conf"))


class TestLogGenerator:
    _var_re = re.compile(r"\$.+?\$")

    def __init__(self, conf_fn, seed=None):
        if seed is None:
            random.seed()
        else:
            random.seed(seed)

        self.conf = configparser.ConfigParser()
        self.conf.read(conf_fn)
        self.term = config.getterm(self.conf, "main", "term")
        self.top_dt, self.end_dt = self.term
        self.d_host = {}
        for group in config.gettuple(self.conf, "main", "host_groups"):
            for host in config.gettuple(self.conf, "main", "group_" + group):
                self.d_host.setdefault(group, []).append(host)

        self.l_event = []
        for event_name in config.gettuple(self.conf, "main", "events"):
            self._generate_event(event_name)

        self.l_log = []
        for event in self.l_event:
            self._generate_log(event)

    def _dt_rand(self, top_dt, end_dt):
        return self._dt_delta_rand(top_dt,
                                   datetime.timedelta(0), end_dt - top_dt)

    @staticmethod
    def _dt_delta_rand(top_dt, durmin, durmax):
        deltasec = 24 * 60 * 60 * (durmax.days - durmin.days) + \
                   durmax.seconds - durmin.seconds
        seconds = datetime.timedelta(seconds=random.randint(0, deltasec - 1))
        return top_dt + durmin + seconds

    @staticmethod
    def rand_uniform(top_dt, end_dt, lambd):
        """Generate a random event that follows uniform distribution of
        LAMBD times a day on the average.

        Args:
            top_dt (datetime.datetime): The start time of generated event.
            end_dt (datetime.datetime): The end time of generated event.
            lambd (float): The average of appearance of generated event per a day.

        Returns:
            list[datetime.datetime]
        """
        ret = []
        total_dt = (end_dt - top_dt)
        avtimes = 1.0 * lambd * total_dt.total_seconds() / (24 * 60 * 60)
        times = int(np.random.poisson(avtimes))
        for i in range(times):
            deltasec = int(total_dt.total_seconds() * random.random())
            dt = top_dt + datetime.timedelta(seconds=deltasec)
            ret.append(dt)
        return ret

    @classmethod
    def _rand_exp(cls, top_dt, end_dt, lambd):
        """Generate a random event that follows Poisson process.
        The event interval matches exponential distribution of
        LAMBD times a day on the average.

        Args:
            top_dt (datetime.datetime): The start time of generated event.
            end_dt (datetime.datetime): The end time of generated event.
            lambd (float): The average of appearance of generated event per a day.

        Yields:
            datetime.datetime
        """
        temp_dt = top_dt
        temp_dt = cls._rand_next_exp(temp_dt, lambd)
        while temp_dt < end_dt:
            yield temp_dt
            temp_dt = cls._rand_next_exp(temp_dt, lambd)

    @staticmethod
    def _rand_next_exp(dt, lambd):
        return dt + datetime.timedelta(seconds=1) * int(
            24 * 60 * 60 * random.expovariate(lambd))

    def _generate_event(self, event_name):

        section = "event_" + event_name

        def _recur(dt, host, event_name):
            if not self.conf.has_option(section, "recurrence"):
                return
            if self.conf.getboolean(section, "recurrence"):
                if random.random() < self.conf.getfloat(section, "recur_p"):
                    durmin = config.getdur(self.conf, section, "recur_dur_min")
                    durmax = config.getdur(self.conf, section, "recur_dur_max")
                    new_dt = self._dt_delta_rand(dt, durmin, durmax)
                    _add_event(new_dt, host, event_name)

        def _add_event(dt, host, event_name):
            info = {}
            if self.conf.has_option(section, "info"):
                for i in config.gettuple(self.conf, section, "info"):
                    if i == "ifname":
                        info[i] = random.choice(
                            config.gettuple(self.conf, section, "ifname"))
                    elif i == "user":
                        info[i] = random.choice(
                            config.gettuple(self.conf, section, "user"))
            self.l_event.append((dt, host, event_name, info))
            _recur(dt, host, event_name)

        for group in config.gettuple(self.conf, section, "groups"):
            for host in self.d_host[group]:
                occ = self.conf.get(section, "occurrence")
                if occ == "random_uniform":
                    freq = self.conf.getfloat(section, "frequency")
                    for dt in self.rand_uniform(self.top_dt, self.end_dt,
                                                freq):
                        _add_event(dt, host, event_name)
                elif occ in ("random", "random_exp"):
                    freq = self.conf.getfloat(section, "frequency")
                    for dt in self._rand_exp(self.top_dt, self.end_dt, freq):
                        _add_event(dt, host, event_name)
                elif occ == "hourly":
                    dursec = 60 * 60
                    first_dt = self.top_dt + datetime.timedelta(
                        seconds=random.randint(0, dursec - 1))
                    now_dt = first_dt
                    while now_dt < self.end_dt:
                        _add_event(now_dt, host, event_name)
                        now_dt += datetime.timedelta(seconds=dursec)
                elif occ == "daily":
                    dursec = 24 * 60 * 60
                    first_dt = self.top_dt + datetime.timedelta(
                        seconds=random.randint(0, dursec - 1))
                    now_dt = first_dt
                    while now_dt < self.end_dt:
                        _add_event(now_dt, host, event_name)
                        now_dt += datetime.timedelta(seconds=dursec)

    def _generate_log(self, event):
        dt = event[0]
        host = event[1]
        event_name = event[2]
        info = event[3]
        for log_name in config.gettuple(self.conf,
                                        "event_" + event_name, "logs"):
            section = "log_" + log_name
            mode = self.conf.get(section, "mode")
            form = self.conf.get(section, "format")

            mes = form
            while True:
                match = self._var_re.search(mes)
                if match is None:
                    break
                var_type = match.group().strip("$")
                if var_type in info.keys():
                    var_string = info[var_type]
                elif var_type == "pid":
                    var_string = str(random.randint(1, 65535))
                elif var_type == "host":
                    var_string = host
                else:
                    raise ValueError
                mes = "".join((mes[:match.start()] + var_string +
                               mes[match.end():]))

            if mode == "each":
                self.l_log.append((dt, host, mes))
            elif mode == "delay_rand":
                delay_min = config.getdur(self.conf, section, "delay_min")
                delay_max = config.getdur(self.conf, section, "delay_max")
                log_dt = self._dt_delta_rand(dt, delay_min, delay_max)
                self.l_log.append((log_dt, host, mes))
            elif mode == "drop_rand":
                drop_p = self.conf.getfloat(section, "drop_p")
                if random.random() > drop_p:
                    self.l_log.append((dt, host, mes))
            elif mode == "other_host_rand":
                l_host = []
                for t_group in config.gettuple(self.conf, section, "groups"):
                    for t_host in self.d_host[t_group]:
                        if not t_host == host:
                            l_host.append(t_host)
                self.l_log.append((dt, random.choice(l_host), mes))

    def dump_log(self, output):
        l_line = sorted(self.l_log, key=lambda x: x[0])
        if output is None:
            for line in l_line:
                print(" ".join((line[0].strftime("%Y-%m-%d %H:%M:%S"),
                                line[1], line[2])))
        else:
            with open(output, 'w') as f:
                for line in l_line:
                    f.write(" ".join((line[0].strftime("%Y-%m-%d %H:%M:%S"),
                                      line[1], line[2])) + "\n")


def generate_testdata(fn=None, output=None, seed=None):
    if fn is None:
        fn = DEFAULT_CONFIG
    tlg = TestLogGenerator(fn, seed)
    tlg.dump_log(output)
