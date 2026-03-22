#[derive(Debug, Clone, Copy)]
pub struct LeaderSimParams {
    pub rounds: u32,
    pub leader_failure_prob: f64,
    pub seed: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct LeaderSimResult {
    pub rounds: u32,
    pub committed: u32,
}

pub fn simulate_leader_based(params: LeaderSimParams) -> LeaderSimResult {
    let mut rng = XorShift64::new(params.seed);
    let mut committed = 0u32;

    for _ in 0..params.rounds {
        let r = rng.next_f64();
        let leader_failed = r < params.leader_failure_prob;
        if !leader_failed {
            committed += 1;
        }
    }

    LeaderSimResult {
        rounds: params.rounds,
        committed,
    }
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        let x = self.next_u64();
        let mantissa = x >> 11;
        (mantissa as f64) / ((1u64 << 53) as f64)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RaftSimParams {
    pub nodes: usize,
    pub ticks: u64,
    pub seed: u64,
    pub election_timeout_min: u64,
    pub election_timeout_max: u64,
    pub heartbeat_interval: u64,
    pub network_delay_min: u64,
    pub network_delay_max: u64,
    pub client_request_prob: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RaftSimResult {
    pub nodes: usize,
    pub ticks: u64,
    pub elections: u64,
    pub leader_changes: u64,
    pub committed_entries: u64,
}

pub fn simulate_raft(params: RaftSimParams) -> RaftSimResult {
    let mut sim = RaftSim::new(params);
    sim.run()
}

#[derive(Debug, Clone)]
enum Role {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone)]
struct Node {
    role: Role,
    term: u64,
    voted_for: Option<usize>,
    leader_id: Option<usize>,
    election_deadline: u64,
    last_heartbeat_tick: u64,
    log_len: u64,
    commit_index: u64,
    votes_received: u64,
    match_index: Vec<u64>,
}

#[derive(Debug, Clone)]
enum Msg {
    RequestVote {
        term: u64,
        candidate: usize,
        last_log_index: u64,
    },
    VoteResponse {
        term: u64,
        voter: usize,
        granted: bool,
    },
    AppendEntries {
        term: u64,
        leader: usize,
        leader_log_len: u64,
        leader_commit: u64,
    },
    AppendEntriesResponse {
        term: u64,
        follower: usize,
        success: bool,
        match_index: u64,
    },
}

#[derive(Debug, Clone, Copy)]
struct AppendEntriesFields {
    term: u64,
    leader: usize,
    leader_log_len: u64,
    leader_commit: u64,
}

#[derive(Debug, Clone, Copy)]
struct AppendEntriesResponseFields {
    term: u64,
    follower: usize,
    success: bool,
    match_index: u64,
}

#[derive(Debug, Clone)]
struct Envelope {
    from: usize,
    to: usize,
    msg: Msg,
}

struct RaftSim {
    params: RaftSimParams,
    rng: XorShift64,
    nodes: Vec<Node>,
    queue: Vec<Vec<Envelope>>,
    elections: u64,
    leader_changes: u64,
    committed_entries: u64,
    leader: Option<usize>,
}

impl RaftSim {
    fn new(params: RaftSimParams) -> Self {
        let nodes = (0..params.nodes)
            .map(|_| Node {
                role: Role::Follower,
                term: 0,
                voted_for: None,
                leader_id: None,
                election_deadline: 0,
                last_heartbeat_tick: 0,
                log_len: 0,
                commit_index: 0,
                votes_received: 0,
                match_index: vec![0; params.nodes],
            })
            .collect::<Vec<_>>();

        let queue_len = (params.ticks + params.network_delay_max + 2) as usize;

        let mut sim = Self {
            params,
            rng: XorShift64::new(params.seed),
            nodes,
            queue: vec![Vec::new(); queue_len],
            elections: 0,
            leader_changes: 0,
            committed_entries: 0,
            leader: None,
        };

        for id in 0..sim.params.nodes {
            let deadline = sim.next_election_deadline(0);
            sim.nodes[id].election_deadline = deadline;
        }

        sim
    }

    fn run(&mut self) -> RaftSimResult {
        for tick in 0..self.params.ticks {
            self.deliver(tick);
            self.tick_nodes(tick);
        }

        RaftSimResult {
            nodes: self.params.nodes,
            ticks: self.params.ticks,
            elections: self.elections,
            leader_changes: self.leader_changes,
            committed_entries: self.committed_entries,
        }
    }

    fn deliver(&mut self, tick: u64) {
        let slot = tick as usize;
        if slot >= self.queue.len() {
            return;
        }
        let envelopes = std::mem::take(&mut self.queue[slot]);
        for env in envelopes {
            let out = self.handle_message(tick, env.to, env.from, env.msg);
            for e in out {
                self.schedule(tick, e);
            }
        }
    }

    fn tick_nodes(&mut self, tick: u64) {
        let mut outgoing = Vec::new();

        for id in 0..self.params.nodes {
            let role = self.nodes[id].role.clone();
            match role {
                Role::Leader => {
                    if tick.saturating_sub(self.nodes[id].last_heartbeat_tick)
                        >= self.params.heartbeat_interval
                    {
                        self.nodes[id].last_heartbeat_tick = tick;
                        outgoing.extend(self.broadcast_append_entries(tick, id));
                    }

                    if self.rng.next_f64() < self.params.client_request_prob {
                        self.nodes[id].log_len += 1;
                        outgoing.extend(self.broadcast_append_entries(tick, id));
                    }
                }
                Role::Follower | Role::Candidate => {
                    if tick >= self.nodes[id].election_deadline {
                        outgoing.extend(self.start_election(tick, id));
                    }
                }
            }
        }

        for e in outgoing {
            self.schedule(tick, e);
        }
    }

    fn start_election(&mut self, tick: u64, candidate: usize) -> Vec<Envelope> {
        self.elections += 1;
        let term = self.nodes[candidate].term + 1;
        self.nodes[candidate].term = term;
        self.nodes[candidate].role = Role::Candidate;
        self.nodes[candidate].voted_for = Some(candidate);
        self.nodes[candidate].votes_received = 1;
        self.nodes[candidate].leader_id = None;
        self.nodes[candidate].election_deadline = self.next_election_deadline(tick);

        let last_log_index = self.nodes[candidate].log_len;

        let mut out = Vec::new();
        for to in 0..self.params.nodes {
            if to == candidate {
                continue;
            }
            out.push(Envelope {
                from: candidate,
                to,
                msg: Msg::RequestVote {
                    term,
                    candidate,
                    last_log_index,
                },
            });
        }
        out
    }

    fn become_leader(&mut self, tick: u64, leader: usize) -> Vec<Envelope> {
        if self.leader != Some(leader) {
            self.leader_changes += 1;
        }
        self.leader = Some(leader);
        self.nodes[leader].role = Role::Leader;
        self.nodes[leader].leader_id = Some(leader);
        self.nodes[leader].last_heartbeat_tick = tick;
        self.nodes[leader].match_index = vec![0; self.params.nodes];
        self.nodes[leader].match_index[leader] = self.nodes[leader].log_len;
        self.broadcast_append_entries(tick, leader)
    }

    fn broadcast_append_entries(&mut self, _tick: u64, leader: usize) -> Vec<Envelope> {
        let term = self.nodes[leader].term;
        let leader_log_len = self.nodes[leader].log_len;
        let leader_commit = self.nodes[leader].commit_index;

        let mut out = Vec::new();
        for to in 0..self.params.nodes {
            if to == leader {
                continue;
            }
            out.push(Envelope {
                from: leader,
                to,
                msg: Msg::AppendEntries {
                    term,
                    leader,
                    leader_log_len,
                    leader_commit,
                },
            });
        }
        out
    }

    fn handle_message(&mut self, tick: u64, to: usize, from: usize, msg: Msg) -> Vec<Envelope> {
        match msg {
            Msg::RequestVote {
                term,
                candidate,
                last_log_index,
            } => self.on_request_vote(tick, to, from, term, candidate, last_log_index),
            Msg::VoteResponse {
                term,
                voter,
                granted,
            } => self.on_vote_response(tick, to, from, term, voter, granted),
            Msg::AppendEntries {
                term,
                leader,
                leader_log_len,
                leader_commit,
            } => self.on_append_entries(
                tick,
                to,
                from,
                AppendEntriesFields {
                    term,
                    leader,
                    leader_log_len,
                    leader_commit,
                },
            ),
            Msg::AppendEntriesResponse {
                term,
                follower,
                success,
                match_index,
            } => self.on_append_entries_response(
                tick,
                to,
                from,
                AppendEntriesResponseFields {
                    term,
                    follower,
                    success,
                    match_index,
                },
            ),
        }
    }

    fn on_request_vote(
        &mut self,
        tick: u64,
        to: usize,
        _from: usize,
        term: u64,
        candidate: usize,
        last_log_index: u64,
    ) -> Vec<Envelope> {
        if term > self.nodes[to].term {
            self.nodes[to].term = term;
            self.nodes[to].role = Role::Follower;
            self.nodes[to].voted_for = None;
            self.nodes[to].leader_id = None;
        }

        let mut granted = false;
        if term == self.nodes[to].term {
            let can_vote =
                self.nodes[to].voted_for.is_none() || self.nodes[to].voted_for == Some(candidate);
            let up_to_date = last_log_index >= self.nodes[to].log_len;
            if can_vote && up_to_date {
                granted = true;
                self.nodes[to].voted_for = Some(candidate);
                self.nodes[to].election_deadline = self.next_election_deadline(tick);
            }
        }

        vec![Envelope {
            from: to,
            to: candidate,
            msg: Msg::VoteResponse {
                term: self.nodes[to].term,
                voter: to,
                granted,
            },
        }]
    }

    fn on_vote_response(
        &mut self,
        tick: u64,
        to: usize,
        _from: usize,
        term: u64,
        _voter: usize,
        granted: bool,
    ) -> Vec<Envelope> {
        if term > self.nodes[to].term {
            self.nodes[to].term = term;
            self.nodes[to].role = Role::Follower;
            self.nodes[to].voted_for = None;
            self.nodes[to].leader_id = None;
            self.nodes[to].votes_received = 0;
            return Vec::new();
        }

        if !matches!(self.nodes[to].role, Role::Candidate) {
            return Vec::new();
        }

        if term != self.nodes[to].term {
            return Vec::new();
        }

        if granted {
            self.nodes[to].votes_received += 1;
            let majority = (self.params.nodes as u64 / 2) + 1;
            if self.nodes[to].votes_received >= majority {
                return self.become_leader(tick, to);
            }
        }

        Vec::new()
    }

    fn on_append_entries(
        &mut self,
        tick: u64,
        to: usize,
        _from: usize,
        msg: AppendEntriesFields,
    ) -> Vec<Envelope> {
        if msg.term < self.nodes[to].term {
            return vec![Envelope {
                from: to,
                to: msg.leader,
                msg: Msg::AppendEntriesResponse {
                    term: self.nodes[to].term,
                    follower: to,
                    success: false,
                    match_index: self.nodes[to].log_len,
                },
            }];
        }

        if msg.term > self.nodes[to].term {
            self.nodes[to].term = msg.term;
            self.nodes[to].voted_for = None;
        }

        self.nodes[to].role = Role::Follower;
        self.nodes[to].leader_id = Some(msg.leader);
        self.nodes[to].election_deadline = self.next_election_deadline(tick);
        self.nodes[to].log_len = msg.leader_log_len;
        self.nodes[to].commit_index = self.nodes[to]
            .commit_index
            .max(msg.leader_commit)
            .min(msg.leader_log_len);

        vec![Envelope {
            from: to,
            to: msg.leader,
            msg: Msg::AppendEntriesResponse {
                term: self.nodes[to].term,
                follower: to,
                success: true,
                match_index: self.nodes[to].log_len,
            },
        }]
    }

    fn on_append_entries_response(
        &mut self,
        _tick: u64,
        to: usize,
        _from: usize,
        msg: AppendEntriesResponseFields,
    ) -> Vec<Envelope> {
        if !matches!(self.nodes[to].role, Role::Leader) {
            return Vec::new();
        }
        if msg.term > self.nodes[to].term {
            self.nodes[to].term = msg.term;
            self.nodes[to].role = Role::Follower;
            self.nodes[to].leader_id = None;
            self.leader = None;
            return Vec::new();
        }
        if msg.term != self.nodes[to].term {
            return Vec::new();
        }
        if !msg.success {
            return Vec::new();
        }

        self.nodes[to].match_index[msg.follower] = msg.match_index;

        let mut replicated = self.nodes[to].match_index.clone();
        replicated.sort_unstable();
        let majority_idx = self.params.nodes / 2;
        let new_commit = replicated[majority_idx];
        if new_commit > self.nodes[to].commit_index {
            let delta = new_commit - self.nodes[to].commit_index;
            self.nodes[to].commit_index = new_commit;
            self.committed_entries = self.committed_entries.saturating_add(delta);
        }

        Vec::new()
    }

    fn schedule(&mut self, tick: u64, env: Envelope) {
        let delay = self.sample_delay();
        let deliver_at = tick.saturating_add(delay);
        let idx = deliver_at as usize;
        if idx < self.queue.len() {
            self.queue[idx].push(env);
        }
    }

    fn sample_delay(&mut self) -> u64 {
        if self.params.network_delay_min >= self.params.network_delay_max {
            return self.params.network_delay_min;
        }
        let span = self.params.network_delay_max - self.params.network_delay_min + 1;
        (self.rng.next_u64() % span) + self.params.network_delay_min
    }

    fn next_election_deadline(&mut self, now_tick: u64) -> u64 {
        let min = self.params.election_timeout_min;
        let max = self.params.election_timeout_max.max(min);
        let timeout = if min == max {
            min
        } else {
            let span = max - min + 1;
            (self.rng.next_u64() % span) + min
        };
        now_tick.saturating_add(timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let params = LeaderSimParams {
            rounds: 1000,
            leader_failure_prob: 0.2,
            seed: 123,
        };
        let a = simulate_leader_based(params);
        let b = simulate_leader_based(params);
        assert_eq!(a.committed, b.committed);
    }

    #[test]
    fn failure_prob_affects_commits() {
        let base = LeaderSimParams {
            rounds: 2000,
            leader_failure_prob: 0.0,
            seed: 42,
        };
        let all = simulate_leader_based(base);
        let some = simulate_leader_based(LeaderSimParams {
            leader_failure_prob: 0.5,
            ..base
        });
        assert_eq!(all.committed, base.rounds);
        assert!(some.committed < all.committed);
    }

    #[test]
    fn raft_sim_deterministic_for_same_seed() {
        let params = RaftSimParams {
            nodes: 5,
            ticks: 2000,
            seed: 7,
            election_timeout_min: 40,
            election_timeout_max: 80,
            heartbeat_interval: 10,
            network_delay_min: 1,
            network_delay_max: 5,
            client_request_prob: 0.2,
        };
        let a = simulate_raft(params);
        let b = simulate_raft(params);
        assert_eq!(a.committed_entries, b.committed_entries);
        assert_eq!(a.leader_changes, b.leader_changes);
    }

    #[test]
    fn raft_sim_commits_some_entries() {
        let params = RaftSimParams {
            nodes: 3,
            ticks: 1500,
            seed: 11,
            election_timeout_min: 30,
            election_timeout_max: 60,
            heartbeat_interval: 10,
            network_delay_min: 1,
            network_delay_max: 3,
            client_request_prob: 0.3,
        };
        let r = simulate_raft(params);
        assert!(r.committed_entries > 0);
        assert!(r.elections > 0);
    }
}
