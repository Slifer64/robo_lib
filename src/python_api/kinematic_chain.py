import numpy as np
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py import urdf
import yaml


class KinematicChain:

    def __init__(self, urdf_filename: str, base_link: str, tool_link: str):

        _, tree = urdf.treeFromFile(urdf_filename)
        self.chain = tree.getChain(base_link, tool_link)

        self.links = [self.chain.getSegment(i) for i in range(self.chain.getNrOfSegments())]

        self.link_id_map = {self.links[i].getName(): i for i in range(len(self.links))}

        self.joints = [link.getJoint() for link in self.links if link.getJoint().getType() != 8]

         # find joint limits
        joint_names = self.get_joint_names()
        joints_yaml = yaml.safe_load(str(URDF.from_xml_file(urdf_filename)))['joints']
        joints_lim = {joint['name']: [joint['limit']['lower'], joint['limit']['upper']]
                      for joint in joints_yaml # if joint['type'] not in ('fixed', 'mimic')}
                      if joint['name'] in joint_names}

        self.joint_lower_lim = []
        self.joint_upper_lim = []
        for v in joints_lim.values():
            self.joint_lower_lim.append(v[0])
            self.joint_upper_lim.append(v[1])

        print(self.joint_lower_lim)
        print(self.joint_upper_lim)

        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

        self.ik_vel_solver = kdl.ChainIkSolverVel_pinv_givens(self.chain)  #, maxiter=200, eps=1e-6)
        self.ik_solver = kdl.ChainIkSolverPos_NR_JL(self.chain,
                                                    self.__to_JntArray(self.joint_lower_lim),
                                                    self.__to_JntArray(self.joint_upper_lim),
                                                    self.fk_solver, self.ik_vel_solver, maxiter=200, eps=1e-6)

    @staticmethod
    def __to_JntArray(a_in) -> kdl.JntArray:

        a_out = kdl.JntArray(len(a_in))
        for i in range(len(a_in)):
            a_out[i] = a_in[i]
        return a_out

    def num_of_joints(self) -> int:
        return self.chain.getNrOfJoints()

    def get_link_names(self):
        return [link.getName() for link in self.links]

    def get_joint_names(self):
        return [joint.getName() for joint in self.joints]
    
    def get_joint_position(self, pose: np.array, q0: np.array):

        n_joints = self.num_of_joints()

        jnt0 = self.__to_JntArray(q0)
        
        kdl_pose = kdl.Frame().Identity()
        for i in range(3):
            kdl_pose.p[i] = pose[i, 3]
            for j in range(3):
                kdl_pose.M[i, j] = pose[i, j]

        jnt = kdl.JntArray(n_joints)
        ret = self.ik_solver.CartToJnt(jnt0, kdl_pose, jnt)

        found_solution = ret >= 0

        q = np.zeros(n_joints)
        # if found_solution:
        for i in range(n_joints):
            q[i] = jnt[i]

        return q, found_solution
    
    def get_pose(self, j_pos: np.array, link_name: str = None) -> np.array:

        q_in = self.__to_JntArray(j_pos)

        if link_name is None:
            seg_ind = -1
        else:
            try:
                seg_ind = self.link_id_map[link_name]
            except KeyError:
                raise RuntimeError('[KinematicChain::get_pose]: requested link' + link_name + 'does not exist...')

        p_out = kdl.Frame()
        ret = self.fk_solver.JntToCart(q_in, p_out, seg_ind)
        if ret < 0:
            raise RuntimeError('[KinematicChain::get_pose]: forward kinematic solver failed...')

        pose = np.eye(4)
        for i in range(3):
            for j in range(4):
                pose[i, j] = p_out[i, j]
        return pose

    def get_position(self, j_pos: np.array, link_name: str = None) -> np.array:
        return self.get_pose(j_pos, link_name)[:3, 3].copy()

    def get_rotm(self, j_pos: np.array, link_name: str = None) -> np.array:
        return self.get_pose(j_pos, link_name)[:3, :3].copy()

    def get_quat(self, j_pos: np.array, link_name: str = None) -> np.array:

        R = self.get_rotm(j_pos, link_name)

        q = np.array([1.0, 0, 0, 0])

        tr = R[0, 0] + R[1, 1] + R[2, 2]

        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2  # S=4*qwh
            q[0] = 0.25 * s
            q[1] = (R[2, 1] - R[1, 2]) / s
            q[2] = (R[0, 2] - R[2, 0]) / s
            q[3] = (R[1, 0] - R[0, 1]) / s

        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s

        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s

        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s

        return q / np.linalg.norm(q)

    def get_jacobian(self, j_pos: np.array) -> np.array:

        n_joints = len(j_pos)
        q_in = self.__to_JntArray(j_pos)

        jac = kdl.Jacobian(n_joints)
        ret = self.jac_solver.JntToJac(q_in, jac)
        if ret < 0:
            raise RuntimeError('[KinematicChain::get_jacobian]: Jacobian solver failed...')

        jacob = np.zeros((6, n_joints))
        for i in range(jacob.shape[0]):
            for j in range(jacob.shape[1]):
                jacob[i, j] = jac[i, j]
        return jacob


if __name__ == '__main__':

    print('============ Kinematic Chain Example ============')

    chain = KinematicChain('../../assets/robot_description/robot_description.urdf',
                           'robot_left_arm_base_link', 'robot_left_arm_zed_link')

    print('*** Joints ***\n', chain.get_joint_names())
    print('\n-----------------------------\n')
    print('*** Links ***\n', chain.get_link_names())
    print('\n-----------------------------\n')
    print('*** Joint Lower limits ***\n', chain.joint_lower_lim)
    print('*** Joint Upper limits ***\n', chain.joint_upper_lim)
    print('\n-----------------------------\n')

    np.random.seed(0)

    n_joints = chain.num_of_joints()
    j_pos = np.array([-0.22, -0.7, 1.63, -4, -1.1, 0.5])  # np.random.rand(n_joints)

    print('==> Forward kinematics:\n')
    pose = chain.get_pose(j_pos)
    pos = chain.get_position(j_pos)
    R = chain.get_rotm(j_pos)
    quat = chain.get_quat(j_pos)
    jacob = chain.get_jacobian(j_pos)

    print('pos:\n', pos)
    print('R:\n', R)
    print('quat:\n', quat)
    print('jacob:\n', jacob)

    print('\n-----------------------------\n')

    print('==> Inverse kinematics:\n')

    j_pos_0 = j_pos + 0.6 * np.random.rand(n_joints)
    j_pos_hat, found_sol = chain.get_joint_position(pose, j_pos_0)
    pose_hat = chain.get_pose(j_pos_hat)
    print('found solution = ', found_sol)
    print('pose:\n', pose)
    print('pose_hat:\n', pose_hat)
    print('j_pos:\n', j_pos)
    print('j_pos_hat:\n', j_pos_hat)

    print('\n-----------------------------\n')

    print('==> Forward kinematics for another link in the chain:')
    link_name = 'robot_left_arm_wrist_2_link'
    pose = chain.get_pose(j_pos, link_name)
    pos = chain.get_position(j_pos, link_name)
    R = chain.get_rotm(j_pos, link_name)
    quat = chain.get_quat(j_pos, link_name)
    print("link: ", link_name)
    print('pos:\n', pos)
    print('R:\n', R)
    print('quat:\n', quat)
    print('=================')




