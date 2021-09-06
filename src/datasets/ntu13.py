import os

import numpy as np

import joblib
import codecs as cs
import codecs

from .dataset import Dataset


# action2motion_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 24, 38]
# change 0 and 8
action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]


class NTU13(Dataset):
    dataname = "ntu13"

    def __init__(self, datapath="data/ntu13", **kwargs):
        self.datapath = datapath
        super().__init__(**kwargs)
        
        motion_desc_file = "ntu_vibe_list.txt"
        
        keep_actions = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        self.num_classes = len(keep_actions)
                
        candi_list = []
        candi_list_desc_name = os.path.join(datapath, motion_desc_file)
        with cs.open(candi_list_desc_name, 'r', 'utf-8') as f:
            for line in f.readlines():
                candi_list.append(line.strip())

        self._joints3d = []
        self._poses = []
        self._num_frames_in_video = []
        self._actions = []
        
        for path in candi_list:
            data_org = joblib.load(os.path.join(datapath, path))
            try:
                vibe_data = data_org[1]
                data_pose = vibe_data["pose"]
                # invert joint 0 and 8 already done in the definition of joints
                data_j3d = vibe_data["joints3d"][:, action2motion_joints]
                # initial pose at origin: on dataset.load()
            except KeyError:
                continue
            action_id = int(path[path.index('A') + 1:-4])
            
            self._poses.append(data_pose)
            self._joints3d.append(data_j3d)
            self._actions.append(action_id)
            self._num_frames_in_video.append(data_pose.shape[0])

        self._actions = np.array(self._actions)
        self._num_frames_in_video = np.array(self._num_frames_in_video)
        
        N = len(self._poses)
        # same set for training and testing
        self._train = np.arange(N)
        self._test = np.arange(N)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = ntu_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        joints3D = self._joints3d[ind][frame_ix]
        return joints3D
        
    def _load_rotvec(self, ind, frame_ix):
        pose = self._poses[ind][frame_ix, :].reshape(-1, 24, 3)
        return pose


ntu_action_enumerator = {
    1: "drink water",
    2: "eat meal or snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat or cap",
    21: "take off a hat or cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call or answer phone",
    29: "playing with phone or tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head or bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze or cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache or heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper) or feeling warm",
    50: "punching or slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap or hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person’s stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person’s ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
}


if __name__ == "__main__":
    dataset = NTU13()
