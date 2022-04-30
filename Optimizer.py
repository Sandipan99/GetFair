from torch.optim import Adam
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Categorical
from fairnessMetrics import disparateMistreatment
import math
from scipy.spatial import ConvexHull


class Trainer:
    def __init__(self, model, cls, data, reward_function,
                 gamma=0.99):  # test_data consists of data/class/sensitive attribute information
        self.model = model
        self.cls = cls
        self.data = data
        self.reward_function = reward_function
        self.gamma = gamma
        #self.learned_coef = np.copy(cls.coef_)
        #self.learned_intercept = np.copy(cls.intercept_)
        self.rewards = []

    def avgReward(self):
        all_rewards = [r[3] for r in self.rewards]
        all_rewards = all_rewards[::-1]
        return sum([(self.gamma ** i) * all_rewards[i] for i in range(len(all_rewards))])

    def train(self, epochs=5000, learning_rate=0.0005):
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        print('starting parameter', self.cls.coef_)
        ts_pred = self.cls.predict(self.data.ts_dt)
        reward = self.reward_function(self.data.ts_s, self.data.ts_c, ts_pred)
        self.rewards.append((self.cls.coef_[0][0], self.cls.coef_[0][1], self.cls.intercept_[0], reward))

        for _ in tqdm(range(epochs)):
            param = np.concatenate((self.cls.coef_, self.cls.intercept_.reshape(1, -1)), axis=1)
            inp = torch.tensor(param, dtype=torch.float32)
            out = self.model(inp)
            self.cls.coef_ = out.detach().numpy()[:, :2]
            self.cls.intercept_ = out.detach().numpy()[:, 2]
            ts_pred = self.cls.predict(self.data.ts_dt)
            score_ = (self.cls.score(self.data.ts_dt, self.data.ts_c))
            # reward function based on the fairness measure..
            reward = torch.tensor(self.reward_function(self.data.ts_s, self.data.ts_c, ts_pred), dtype=torch.float32)
            self.rewards.append((self.cls.coef_[0][0], self.cls.coef_[0][1], self.cls.intercept_[0], reward.item(),
                                 score_))
            discount = self.avgReward()
            loss = torch.sum(torch.log(out) * (reward - discount))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def plotSeparator(self):
        fig, ax = plt.subplots()
        for i in range(self.data.ts_dt.shape[0]):
            if self.data.ts_c[i] == 0:
                if self.data.ts_s[i] == 0:
                    ax.scatter(self.data.ts_dt[i][0], self.data.ts_dt[i][1], c='r', marker='x')
                else:
                    ax.scatter(self.data.ts_dt[i][0], self.data.ts_dt[i][1], c='r', marker='1')
            else:
                if self.data.ts_s[i] == 0:
                    ax.scatter(self.data.ts_dt[i][0], self.data.ts_dt[i][1], c='b', marker='x')
                else:
                    ax.scatter(self.data.ts_dt[i][0], self.data.ts_dt[i][1], c='b', marker='1')

        x = np.linspace(-8, 8)

        for i in [100, 500, 1000, 1500, 2000]:
            w = np.array([[self.rewards[i][0], self.rewards[i][1]]])
            b = np.array([self.rewards[i][2]])

            y = -b[0] / w[0][1] - (w[0][0] * x) / w[0][1]
            plt.plot(x, y, color='k')

        plt.show()

    def clsScore(self):
        return self.cls.score(self.data.ts_dt, self.data.ts_c)

    def clsScoreatK(self, k):
        self.cls.coef_ = np.array([[self.rewards[k][0], self.rewards[k][1]]])
        self.cls.intercept_ = np.array([self.rewards[k][2]])
        return self.score()


class TrainerRNN(Trainer):
    def __init__(self, model, cls, data, reward_function, gamma=0.1):
        super(TrainerRNN, self).__init__(model, cls, data, reward_function, gamma=gamma)
        # super(TrainerRNN, self).__init__(model, cls, data, reward_function, gamma=gamma)

    def train(self, epochs=250, learning_rate=0.009, step_size=0.05, step_size_gamma=0.99):
        '''
        The training procedure runs for the number of epochs updating the optimizer's parameter at each step
        by calculating the reward at each step. The problem is since the model has no idea of the future rewards at
        a given step, it just sticks to one of the actions and maximizes it.
        :param epochs: the number of steps for which the trainer runs
        :param learning_rate: learning rate of the optimizer model
        :param step_size: the size by which we change the parameters of the classifier
        :param step_size_gamma: This determines how the steps size change as we move continue with the training. The
        step_size will get smaller towards the end as we should ideally be close to the best parameters for the
        classifier
        :return:
        '''
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print('starting parameter', self.cls.coef_)
        ts_pred = self.cls.predict(self.data.ts_dt)
        reward = self.reward_function(self.data.ts_s, ts_pred)
        self.rewards.append((self.cls.coef_[0][0], self.cls.coef_[0][1], self.cls.intercept_[0], reward))
        for _ in tqdm(range(epochs)):
            param = np.concatenate((self.cls.coef_, self.cls.intercept_.reshape(1, -1)), axis=1)
            inp = torch.tensor(param, dtype=torch.float32)
            out = self.model(inp)
            #dir = torch.argmax(out, dim=1).detach().numpy()

            # if the classifier predicts 1 we take a small step forward (add a small value)
            # if the classifier predicts 0 we take a small step backward (subtract a small value)
            prob, dir = torch.max(out, dim=1)
            conf = prob.detach().numpy() - 0.5
            dir = dir.detach().numpy()
            self.cls.coef_[0][0] = self.cls.coef_[0][0] - conf[0]*step_size if dir[0] == 1 else \
                self.cls.coef_[0][0] + conf[0]*step_size
            self.cls.coef_[0][1] = self.cls.coef_[0][1] - conf[1]*step_size if dir[1] == 1 else \
                self.cls.coef_[0][1] + conf[1]*step_size
            self.cls.intercept_[0] = self.cls.intercept_[0] - conf[2]*step_size if dir[2] == 1 else \
                self.cls.intercept_[0] + conf[2]*step_size

            # print(self.cls.coef_)
            # print(self.cls.intercept_)
            ts_pred = self.cls.predict(self.data.ts_dt)
            score_ = self.cls.score(self.data.ts_dt, self.data.ts_c)
            reward = torch.tensor(self.reward_function(self.data.ts_s, ts_pred), dtype=torch.float32)
            self.rewards.append((self.cls.coef_[0][0], self.cls.coef_[0][1], self.cls.intercept_[0], reward.item(),
                                 score_))
            # discount = self.avgReward()
            # loss = torch.sum(torch.log(out) * (reward - discount))
            # print(reward.requires_grad)
            #print(out)
            #print(prob)
            loss = -torch.sum(torch.log(prob) * reward)
            #print(-loss)

            if _ % 25 == 0:
                print(f'classifier coefficient - {self.cls.coef_}')
                print(f'classifier intercept - {self.cls.intercept_}')
                print(f'Confidence - {out} ')
                print(f'Score = {score_}')
                print(f'Reward = {reward}')
                #print(out, loss, reward)

            #if _ in range(100,150): # always prefering one action over the other
            #    print(out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # step_size reduces as we reach the optima...
            # if step_size_gamma = 1.0, the step_size remains fixed
            step_size = step_size*step_size_gamma


class TrainerEpisodic(Trainer):
    def __init__(self, model, cls, data, reward_function, classifiertype=None, gamma=0.1, lbda=0.5):
        super(TrainerEpisodic, self).__init__(model, cls, data, reward_function, gamma=gamma)
        self.best_param = None
        self.learned_coef = np.copy(cls.coef_) if classifiertype == 'linear' else np.copy(cls.coefs_[-1])
        self.learned_intercept = np.copy(cls.intercept_) if classifiertype == 'linear' \
            else np.copy(cls.intercepts_[-1])
        self.max_reward = 0
        self.max_acc = 0
        self.all_trajs = []
        self.best_episode = 0
        self.classifiertype = classifiertype
        self.l = lbda

    def reset(self):
        if self.classifiertype == 'linear':
            self.cls.coef_ = np.copy(self.learned_coef)
            self.cls.intercept_ = np.copy(self.learned_intercept)
            self.rewards = []
        elif self.classifiertype == 'neuralnet':
            self.cls.coefs_[-1] = np.copy(self.learned_coef)
            self.cls.intercepts_[-1] = np.copy(self.learned_intercept)
            self.rewards = []

    def update_params(self, conf, action, step_size):
        if self.classifiertype == 'linear':
            for i in range(len(conf) - 1):
                if action[i] == 0:
                    self.cls.coef_[0][i] = self.cls.coef_[0][i] + step_size  # * conf[i]
                elif action[i] == 1:
                    self.cls.coef_[0][i] = self.cls.coef_[0][i] - step_size  # * conf[i]
                else:
                    pass
                # self.cls.coef_[0][i] = self.cls.coef_[0][i] - conf[i] * step_size if action[i] == 1 else \
                #        self.cls.coef_[0][i] + conf[i] * step_size

            if action[-1] == 0:
                self.cls.intercept_[0] = self.cls.intercept_[0] + step_size  # * conf[-1]
            elif action[-1] == 1:
                self.cls.intercept_[0] = self.cls.intercept_[0] - step_size  # * conf[-1]
            else:
                pass

        elif self.classifiertype == 'neuralnet': # only update parameters of the last layer...
            for i in range(len(conf) - 1):
                if action[i] == 0:
                    self.cls.coefs_[-1][i] = self.cls.coefs_[-1][i] + step_size  # * conf[i]
                elif action[i] == 1:
                    self.cls.coefs_[-1][i] = self.cls.coefs_[-1][i] - step_size  # * conf[i]
                else:
                    pass
                # self.cls.coef_[0][i] = self.cls.coef_[0][i] - conf[i] * step_size if action[i] == 1 else \
                #        self.cls.coef_[0][i] + conf[i] * step_size

            if action[-1] == 0:
                self.cls.intercepts_[-1] = self.cls.intercepts_[-1] + step_size  # * conf[-1]
            elif action[-1] == 1:
                self.cls.intercepts_[-1] = self.cls.intercepts_[-1] - step_size  # * conf[-1]
            else:
                pass
        else: # option to add another classifier
            pass

    def get_param(self, classifier=None):
        if self.classifiertype == 'linear':
            return np.concatenate((self.cls.coef_, self.cls.intercept_.reshape(1, -1)), axis=1)
        elif self.classifiertype == 'neuralnet':
            return np.concatenate((self.cls.coefs_[-1].reshape(1, -1), self.cls.intercepts_[-1].reshape(1, -1)), axis=1)
        else: # option to add another classifier
            pass

    def get_step_size(self, index, routine='const'):
        '''
        :param index: Current time point in the episode
        :param routine: const -> stepsize is same irrespective of the time point
                        cyclic -> stepsize is cyclic.. obtained using a cosine function
        :return: the factor by which the step size is to be normalized for the current time point
        '''
        index = index % 360
        if routine == 'const':
            return 1
        elif routine == 'cyclic':
            return abs(math.cos(index*math.pi/180))

    def train(self, episodes=100, max_episode_len=400, learning_rate=0.02, step_size=0.06, step_size_gamma=1.0,
              accuracy_threshold=0.75):
        '''
        The key difference with TrainerRNN class is this model is episodic. We decide on a metaoptimizer model
        and allow it to run it for a one full episode of size episode length. Calculate discounted reward for
        the whole episode and then update the parameters of the optimizer model based on the cumulative reward. What
        was observed in case of TrainerRNN was that it was unstable in the sense that for some episodes it was doing
        well while for others it was not. Another key difference is that instead of a greedy action we will
        probabilistically sample an action.
        :param episodes: The number of episodes to run the training
        :param learning_rate: Learning rate of the optimizer model - we use an RNN based model
        :param step_size: The size by which we change the parameters of the classifier model at each step
        :param step_size_gamma: This determines how the steps size change as we move continue with the training. The
        step_size will get smaller towards the end as we should ideally be close to the best parameters for the
        classifier
        :param accuracy_threshold: if the accuracy goes below this threshold the episode is ended
        '''
        #print(f'discounting factor: {self.gamma}')
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        #if self.classifiertype == 'linear':
        #    print('starting parameter', self.cls.coef_, self.cls.intercept_)
        #else:
        #    print('starting parameter', self.cls.coefs_[-1], self.cls.intercepts_[-1])
        ts_pred = self.cls.predict(self.data.ts_dt)
        reward = self.reward_function(self.data.ts_s, ts_pred, self.data.ts_c)
        #print(f'starting reward: {reward}')  # reward value at the start of the process
        for e in tqdm(range(episodes)):
            rewards = []
            policy_history = torch.Tensor()
            traj = []
            for s in range(max_episode_len): # start of a episode
                param = self.get_param()
                inp = torch.tensor(param, dtype=torch.float32)
                out = self.model(inp)
                #print(out)
                c = Categorical(out)
                action = c.sample()
                policy = torch.sum(c.log_prob(action)).reshape(1, 1)
                action = action.detach().numpy() # sampling action
                conf = np.array([out[i, action[i]].item() for i in range(len(action))])
                #conf = conf - 0.5
                # update parameters of the classifier
                step_size_i = step_size * self.get_step_size(s, routine='cyclic')
                self.update_params(conf, action, step_size_i)
                ts_pred = self.cls.predict(self.data.ts_dt)
                score_ = self.cls.score(self.data.ts_dt, self.data.ts_c)
                # Calculate reward
                r_f = self.reward_function(self.data.ts_s, ts_pred, self.data.ts_c)
                if r_f is None:
                    reward = 0
                else:
                    #print(r_f, score_)
                    reward = self.l*r_f + (1-self.l)*score_
                    #print(reward)
                if self.classifiertype == 'linear':
                    traj.append((score_, reward, np.copy(self.cls.coef_), np.copy(self.cls.intercept_)))
                else:
                    traj.append((score_, reward, np.copy(self.cls.coefs_[-1]), np.copy(self.cls.intercepts_[-1])))

                reward = reward + torch.mean(c.entropy()).detach() #+ score_
                rewards.append(reward)
                policy_history = torch.cat((policy_history, policy), dim=1)
                step_size = step_size * step_size_gamma
                if score_ < accuracy_threshold:
                    break
            # end of episode....
            #print(rewards)
            #print(self.gamma)
            #print(f'length of episode - {s}')
            self.all_trajs.append(traj)
            R = 0
            discounted_rewards = []
            for r in rewards[::-1]:
                R = r + self.gamma*R
                discounted_rewards.insert(0, R)

            discounted_rewards = torch.FloatTensor(discounted_rewards).reshape(1, -1)
            den = torch.std(discounted_rewards, unbiased=False) + 0.0000001
            # by default torch.std returns unbiased estimate of std i.e., the score is normalized by n-1
            # instead of n. So if the episode is of length 1, torch.std returns Nan
            # we hence obtain the biased estimate of std. This occurs when the expected accuracy after
            # bias removal is very close to the original accuracy as one step could lead to an accuracy less
            # than the threshold and hence creating an episode of length 1. Mainly encountered during the
            # the earlier episodes. This could also be handled by lowering the learning rate of the meta
            # optimizer or lowering the step size.
            discounted_rewards = discounted_rewards - torch.mean(discounted_rewards)
            discounted_rewards = discounted_rewards/den
            loss = -torch.sum(torch.mul(policy_history, discounted_rewards))
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if e % 10 == 0 or e == episodes-1:
                #print(f'iteration: {e}')
                #print(f'max reward achieved: {max(rewards)}')
                #if self.classifiertype == 'linear':
                #    print(f'classifier coefficient - {self.cls.coef_}')
                #    print(f'classifier intercept - {self.cls.intercept_}')
                #elif self.classifiertype == 'neuralnet':
                #    print(f'classifier coefficient - {self.cls.coefs_[-1]}')
                #    print(f'classifier intercept - {self.cls.intercepts_[-1]}')
                ts_pred = self.cls.predict(self.data.ts_dt)
                score_ = self.cls.score(self.data.ts_dt, self.data.ts_c)
                reward_ = self.reward_function(self.data.ts_s, ts_pred, self.data.ts_c)
                #print(f'classifier accuracy - {score_}')
                #print(f'disparity - {reward_}')

            self.reset() # reset classifier to original parameters....

    def test(self):
        if self.classifiertype == 'linear':
            self.cls.coef_ = self.best_param[0]
            self.cls.intercept_ = self.best_param[1]
        elif self.classifiertype == 'neuralnet':
            self.cls.coefs_[-1] = self.best_param[0]
            self.cls.intercepts_[-1] = self.best_param[1]
        else:
            pass
        ts_pred = self.cls.predict(self.data.ts_dt)
        score_ = self.cls.score(self.data.ts_dt, self.data.ts_c)
        reward = self.reward_function(self.data.ts_s, ts_pred, self.data.ts_c)
        print(f'max reward: {self.max_reward}')
        print(f'coefficients: {self.best_param[0]}')
        print(f'intercept: {self.best_param[1]}')
        print(f'best accuracy: {score_}, disparity: {reward}, episode: {self.best_episode}')
        print(f'disparate mistreatment: {disparateMistreatment(self.data.ts_s, self.data.ts_c, ts_pred)}')

    def skyline(self, index=None):
        '''
        :param index: if None plot all trajectories else plot the trajectory of a particular index default:None
        :return: None
        '''
        #print(f'all rewards - {len(self.all_trajs)}')
        if index is not None:
            accr_, fairness = zip(*self.all_trajs[index])
            plt.plot(accr_, fairness, linestyle='--', marker='o')
        else: # print 10 best episodes
            #max_reward = 0
            episode_reward = []
            for i in range(len(self.all_trajs)):
                m_r = sorted(self.all_trajs[i], key=lambda x: x[1], reverse=True)[0][1]
                #print(f'max reward for episode - {i} is {m_r}')
                episode_reward.append((i, m_r))
            k = 10
            for ep_, rwd in sorted(episode_reward, key=lambda x: x[1], reverse=True):
                accr_, fairness = zip(*self.all_trajs[ep_])
                plt.plot(accr_, fairness, linestyle='--', marker='o', label=str(ep_))
                k -= 1
                if k == 0:
                    break
        plt.xlabel('accuracy')
        plt.ylabel('fairness')
        plt.legend()
        plt.show()

    def get_best_parameters(self, accuracy_threshold=0.5):
        '''
        Generates a convex hull over the points (accuracy, fairness) to obtain the best fairness-accuracy trade-off
        :return: The best parameter configuration for the classifier for best accurcy-fairness tradeoff
        '''
        points = [] # contains the points for which convex hull is to be obtained
        points_index = [] # contains the index of the selected point so as to retrieve the parameters during inference
        #print(len(self.all_trajs))
        for i in range(len(self.all_trajs)):
            traj = self.all_trajs[i]
            #print(len(traj))
            for j in range(len(traj)):
               acc, reward, _, _ = traj[j]
               if acc >= accuracy_threshold:
                   points.append([acc, reward])
                   points_index.append((i, j))

        hull = ConvexHull(points)
        # return result at different thresholds of frequency
        hull_points = []
        for p in hull.vertices:
            corner = points[p]
            #print(corner)
            i, j = points_index[p][0], points_index[p][1]
            #print(corner, i, j)
            hull_points.append((corner[0], corner[1], self.all_trajs[i][j][2], self.all_trajs[i][j][3]))


        thresholds = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
        best_settings = {}
        ind_ = 0
        max_fairness = 0
        max_acc = 0
        parameter = None
        for acc, fairness, param_0, param_1 in sorted(hull_points, key=lambda x: x[0], reverse=True):
            if ind_ > len(thresholds)-1:
                break
            if acc >= thresholds[ind_]:
                if fairness > max_fairness:
                    max_fairness = fairness
                    parameter = (param_0, param_1)
                    max_acc = acc
            else:
                best_settings[str(thresholds[ind_])] = (max_acc, max_fairness, parameter)
                ind_ += 1
                if ind_ > len(thresholds) - 1:
                    break
                while acc < thresholds[ind_]:
                    ind_ += 1
                    if ind_ > len(thresholds) - 1:
                        break
                max_fairness = fairness
                max_acc = acc
                parameter = (param_0, param_1)

        results_bp = {}

        for key in best_settings:
            print(key)
            #print(best_settings[key][0], best_settings[key][1])
            if self.classifiertype == 'linear':
                self.cls.coef_ = np.copy(best_settings[key][2][0])
                self.cls.intercept_ = np.copy(best_settings[key][2][1])
            elif self.classifiertype == 'neuralnet':
                self.cls.coefs_[-1] = np.copy(best_settings[key][2][0])
                self.cls.intercepts_[-1] = np.copy(best_settings[key][2][1])
            ts_pred = self.cls.predict(self.data.vl_dt)
            score_ = self.cls.score(self.data.vl_dt, self.data.vl_c)
            reward = self.reward_function(self.data.vl_s, ts_pred, self.data.vl_c)
            results_bp[key] = (score_, reward)
            print(score_, reward)
            print('--------------')
        return results_bp

    def get_best_parameters_alt(self, rp=False):
        #thresholds = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
        best_tradeoff = [{'max_fairness': 0, 'accuracy': 0, 'index_i': None, 'index_j': None} for _ in range(8)]

        for i in range(len(self.all_trajs)):
            traj = self.all_trajs[i]
            # print(len(traj))
            for j in range(len(traj)):
                acc, reward, _, _ = traj[j]
                if acc >= 0.9:
                    index = 0
                elif acc >= 0.85:
                    index = 1
                elif acc >= 0.8:
                    index = 2
                elif acc >= 0.75:
                    index = 3
                elif acc >= 0.7:
                    index = 4
                elif acc >= 0.65:
                    index = 5
                elif acc >= 0.6:
                    index = 6
                elif acc >= 0.55:
                    index = 7
                else:
                    index = None

                if index is not None:
                    if reward > best_tradeoff[index]['max_fairness']:
                        best_tradeoff[index]['max_fairness'] = reward
                        best_tradeoff[index]['index_i'] = i
                        best_tradeoff[index]['index_j'] = j
                        best_tradeoff[index]['accuracy'] = acc

        results_bp = {}

        for i in range(8):
            t_i, t_j = best_tradeoff[i]['index_i'], best_tradeoff[i]['index_j']
            if t_i is not None and t_j is not None:
                if self.classifiertype == 'linear':
                    self.cls.coef_ = np.copy(self.all_trajs[t_i][t_j][2])
                    self.cls.intercept_ = np.copy(self.all_trajs[t_i][t_j][3])
                elif self.classifiertype == 'neuralnet':
                    self.cls.coefs_[-1] = np.copy(self.all_trajs[t_i][t_j][2])
                    self.cls.intercepts_[-1] = np.copy(self.all_trajs[t_i][t_j][3])
                else:
                    pass

                coef = np.copy(self.all_trajs[t_i][t_j][2])
                interp = np.copy(self.all_trajs[t_i][t_j][3])
                ts_pred = self.cls.predict(self.data.vl_dt)
                score_ = self.cls.score(self.data.vl_dt, self.data.vl_c)
                reward = self.reward_function(self.data.vl_s, ts_pred, self.data.vl_c)
                if rp:
                    results_bp[i] = (score_, reward, coef, interp)
                else:
                    results_bp[i] = (score_, reward)
                #print(score_, reward)
                #print('--------------')
        return results_bp

######################################################################


class TrainerEpisodicMultiple(TrainerEpisodic):
    def __init__(self, model, cls, data, reward_function, classifiertype=None, gamma=0.1, lbda=1.0):
        super(TrainerEpisodicMultiple, self).__init__(model, cls, data, reward_function,
                                                      classifiertype=classifiertype,
                                                      gamma=gamma, lbda=lbda)

    def compute_reward(self, ts_pred, metric_1, metric_2):
        r_1 = metric_1(self.data.ts_s, ts_pred, self.data.ts_c)
        r_2 = metric_2(self.data.ts_s, ts_pred, self.data.ts_c)
        return r_1, r_2

    def train(self, episodes=100, max_episode_len=400, learning_rate=0.02, step_size=0.06, step_size_gamma=1.0,
              accuracy_threshold=0.75):
        '''
        overriding the train function
        '''
        # print(f'discounting factor: {self.gamma}')
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        ts_pred = self.cls.predict(self.data.ts_dt)
        metric_1 = self.reward_function[0]
        metric_2 = self.reward_function[1]
        r_1, r_2 = self.compute_reward(ts_pred, metric_1, metric_2)
        reward = 0.5*(r_1 + r_2)
        print(f'starting reward: {reward}, {r_1}, {r_2}')  # reward value at the start of the process
        for e in tqdm(range(episodes)):
            rewards = []
            policy_history = torch.Tensor()
            traj = []
            for s in range(max_episode_len):  # start of a episode
                param = self.get_param()
                inp = torch.tensor(param, dtype=torch.float32)
                out = self.model(inp)
                # print(out)
                c = Categorical(out)
                action = c.sample()
                policy = torch.sum(c.log_prob(action)).reshape(1, 1)
                action = action.detach().numpy()  # sampling action
                conf = np.array([out[i, action[i]].item() for i in range(len(action))])
                # conf = conf - 0.5
                # update parameters of the classifier
                step_size_i = step_size * self.get_step_size(s, routine='cyclic')
                self.update_params(conf, action, step_size_i)
                ts_pred = self.cls.predict(self.data.ts_dt)
                score_ = self.cls.score(self.data.ts_dt, self.data.ts_c)
                # Calculate reward
                r_1, r_2 = self.compute_reward(ts_pred, metric_1, metric_2)
                if r_1 is None or r_2 is None:
                    reward = 0
                else:
                    # print(r_f, score_)
                    r_f = 0.5*(r_1 + r_2)
                    reward = self.l * r_f + (1 - self.l) * score_
                    # print(reward)
                if self.classifiertype == 'linear':
                    traj.append((score_, reward, np.copy(self.cls.coef_), np.copy(self.cls.intercept_)))
                else:
                    traj.append((score_, reward, np.copy(self.cls.coefs_[-1]), np.copy(self.cls.intercepts_[-1])))

                reward = reward + torch.mean(c.entropy()).detach()  # + score_
                rewards.append(reward)
                policy_history = torch.cat((policy_history, policy), dim=1)
                step_size = step_size * step_size_gamma
                if score_ < accuracy_threshold:
                    break

            self.all_trajs.append(traj)
            R = 0
            discounted_rewards = []
            for r in rewards[::-1]:
                R = r + self.gamma * R
                discounted_rewards.insert(0, R)

            discounted_rewards = torch.FloatTensor(discounted_rewards).reshape(1, -1)
            den = torch.std(discounted_rewards, unbiased=False) + 0.0000001

            discounted_rewards = discounted_rewards - torch.mean(discounted_rewards)
            discounted_rewards = discounted_rewards / den
            loss = -torch.sum(torch.mul(policy_history, discounted_rewards))
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.reset()  # reset classifier to original parameters....

    def get_best_parameters_alt(self, rp=False):
        # thresholds = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
        best_tradeoff = [{'max_fairness': 0, 'accuracy': 0, 'index_i': None, 'index_j': None} for _ in range(8)]
        metric_1, metric_2 = self.reward_function
        for i in range(len(self.all_trajs)):
            traj = self.all_trajs[i]
            # print(len(traj))
            for j in range(len(traj)):
                acc, reward, _, _ = traj[j]
                if acc >= 0.9:
                    index = 0
                elif acc >= 0.85:
                    index = 1
                elif acc >= 0.8:
                    index = 2
                elif acc >= 0.75:
                    index = 3
                elif acc >= 0.7:
                    index = 4
                elif acc >= 0.65:
                    index = 5
                elif acc >= 0.6:
                    index = 6
                elif acc >= 0.55:
                    index = 7
                else:
                    index = None

                if index is not None:
                    if reward > best_tradeoff[index]['max_fairness']:
                        best_tradeoff[index]['max_fairness'] = reward
                        best_tradeoff[index]['index_i'] = i
                        best_tradeoff[index]['index_j'] = j
                        best_tradeoff[index]['accuracy'] = acc

        results_bp = {}

        for i in range(8):
            t_i, t_j = best_tradeoff[i]['index_i'], best_tradeoff[i]['index_j']
            if t_i is not None and t_j is not None:
                if self.classifiertype == 'linear':
                    self.cls.coef_ = np.copy(self.all_trajs[t_i][t_j][2])
                    self.cls.intercept_ = np.copy(self.all_trajs[t_i][t_j][3])
                elif self.classifiertype == 'neuralnet':
                    self.cls.coefs_[-1] = np.copy(self.all_trajs[t_i][t_j][2])
                    self.cls.intercepts_[-1] = np.copy(self.all_trajs[t_i][t_j][3])
                else:
                    pass

                coef = np.copy(self.all_trajs[t_i][t_j][2])
                interp = np.copy(self.all_trajs[t_i][t_j][3])
                ts_pred = self.cls.predict(self.data.vl_dt)
                score_ = self.cls.score(self.data.vl_dt, self.data.vl_c)
                r_1, r_2 = self.compute_reward(ts_pred, metric_1, metric_2)
                reward = (r_1, r_2)
                if rp:
                    results_bp[i] = (score_, reward, coef, interp)
                else:
                    results_bp[i] = (score_, reward)
                # print(score_, reward)
                # print('--------------')
        return results_bp