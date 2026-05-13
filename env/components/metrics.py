import numpy as np

class RacingMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.overtaking_suc = []
        self.overtook_suc = []
        self.overtaking_cumsum = [0]
        self.overtook_cumsum = [0]
        self.leader_history = []
        self.distances = []
        self.leader = 0  # 0: TV, 1: EV
        self.first_overtake = 0

    def update(self, step_idx, state_ev, state_tv, theta_ev, theta_tv):
        # 1. Distance
        dist = float(np.hypot(state_ev[0] - state_tv[0], state_ev[1] - state_tv[1]))
        self.distances.append(dist)

        # 2. Leader check
        current_overtaking = 0
        current_overtook = 0
        
        if (self.leader == 0) and (theta_ev > theta_tv):
            # EV overtakes TV
            current_overtaking = 1
            self.leader = 1
            if self.first_overtake == 0:
                self.first_overtake = step_idx
        elif (self.leader == 1) and (theta_tv > theta_ev):
            # TV overtakes EV
            current_overtook = 1
            self.leader = 0

        self.overtaking_suc.append(current_overtaking)
        self.overtook_suc.append(current_overtook)
        self.leader_history.append(self.leader)
        
        # 3. Cumulative overtaking counts
        self.overtaking_cumsum.append(self.overtaking_cumsum[-1] + current_overtaking)
        self.overtook_cumsum.append(self.overtook_cumsum[-1] + current_overtook)

        return self.leader

    def get_info(self, lap, solver_status):
        """Racing.py's info dictionary generation"""
        return {
            'overtaking_suc': self.overtaking_suc,
            'overtook_suc': self.overtook_suc,
            'leader': self.leader,
            'first_overtake': self.first_overtake,
            'lap': lap,
            'status': solver_status
        }