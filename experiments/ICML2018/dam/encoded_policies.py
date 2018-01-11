from trlib.policies.policy import Policy

class EncodedPolicy(Policy):
    
    def sample_action(self, state):
        day = state[1]
        
        if day < 60:
            return 5
        elif day < 120:
            return 4
        elif day < 180:
            return 4
        elif day < 240:
            return 3
        elif day < 300:
            return 4
        else:
            return 5
        
class EncodedPolicy1(Policy):
    
    def sample_action(self, state):
        day = state[1]
        
        if day < 60:
            return 3
        elif day < 120:
            return 3
        elif day < 180:
            return 6
        elif day < 240:
            return 6
        elif day < 300:
            return 2
        else:
            return 1
        
class EncodedPolicy2(Policy):
    
    def sample_action(self, state):
        day = state[1]
        
        if day < 60:
            return 6
        elif day < 120:
            return 5
        elif day < 180:
            return 2
        elif day < 240:
            return 3
        elif day < 300:
            return 3
        else:
            return 3
        
class EncodedPolicy3(Policy):
    
    def sample_action(self, state):
        day = state[1]
        
        if day < 60:
            return 5
        elif day < 120:
            return 4
        elif day < 180:
            return 2
        elif day < 240:
            return 3
        elif day < 300:
            return 4
        else:
            return 5