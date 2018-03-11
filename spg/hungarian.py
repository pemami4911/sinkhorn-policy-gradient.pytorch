# Python implementation of post at
# https://www.topcoder.com/community/data-science/data-science-tutorials/assignment-problem-and-hungarian-algorithm/
import numpy as np
import pdb

__all__ = ['Hungarian']


class Hungarian:
    def __init__(self):
        self.max_match = 0
        self.cost = None
        self.n = 0
        self.xy = None
        self.yx = None
        self.S = None
        self.T = None
        self.slack = None
        self.slackx = None
        self.prev = None
        self.lx = None
        self.ly = None
        self.iters = 0


    def __reset(self, cost):
        if np.isnan(np.sum(cost)):
            return False
        self.iters = 0
        # if any entries < 0, clamp to 0
        cost[cost < 0] = 0.
        nx, ny = np.shape(cost)
        assert nx == ny

        self.n = nx  # n of workers
        self.max_match = 0  # n of jobs
        self.xy = [-1 for _ in range(self.n)]  # vertex that is matched with x
        self.yx = [-1 for _ in range(self.n)]  # vertex that is matched with y
        self.S = [False for _ in range(self.n)]  # sets S and T in algorithm
        self.T = [False for _ in range(self.n)]
        self.slack = np.zeros(self.n)  # amount by which sum of labels exceed edge weights
        self.slackx = np.zeros(self.n)  # such a vertex that l(slackx[y]) + l(y) - w(slackx[y],y) = slack[y]
        self.prev = -1 * np.ones(self.n)  # array for memorizing alternating paths
        self.cost = cost

        # init labels
        self.lx = np.zeros(self.n)
        self.ly = np.zeros(self.n)  # labels of Y parts
        for x in range(self.n):
            for y in range(self.n):
                self.lx[x] = max(self.lx[x], self.cost[x, y])
        return True
    
    def __update_labels(self):
        delta = np.inf
        for y in range(self.n):  # calculate delta
            if not self.T[y]:
                delta = min(delta, self.slack[y])
        for x in range(self.n):  # update X labels
            if self.S[x]:
                self.lx[x] -= delta
        for y in range(self.n):  # update Y labels
            if self.T[y]:
                self.ly[y] += delta
        for y in range(self.n):  # update slack array
            if not self.T[y]:
                self.slack[y] -= delta

    def __add_to_tree(self, x, prevx):
        """
        args:
            x: current vertex
            prevx: vertex from X before x in the alternating path,
            so we add edges (prevx, xy[x]), (xy[x], x)
        """
        x = int(x)
        prevx = int(prevx)
        self.S[x] = True
        self.prev[x] = prevx
        for y in range(self.n):
            if self.lx[x] + self.ly[y] - self.cost[x, y] < self.slack[y]:
                self.slack[y] = self.lx[x] + self.ly[y] - self.cost[x, y]
                self.slackx[y] = x

    def __augment(self):
        if self.max_match == self.n:
            return

        root = 0
        x = y = root
        q = [0 for _ in range(self.n)]
        wr = 0
        rd = 0
        # reset
        self.S = [False for _ in range(self.n)]
        self.T = [False for _ in range(self.n)]
        self.prev = [-1 for _ in range(self.n)]

        # finding root of the tree
        for x in range(self.n):
            if self.xy[x] == -1:
                q[wr] = root = x
                wr += 1
                self.prev[x] = -2
                self.S[x] = 1
                break

        for y in range(self.n):
            self.slack[y] = self.lx[root] + self.ly[y] - self.cost[root, y]
            self.slackx[y] = root

        while True:
            self.iters += 1
            if self.iters > 10000:
                print(self.cost)
                pdb.set_trace()

            while rd < wr:  # building tree with bfs cycle
                x = q[rd]  # current vertex from X part
                rd += 1
                while y < self.n:  # iterate through all edges in equality graph
                    if (self.cost[x, y] == self.lx[x] + self.ly[y]) and not self.T[y]:
                        if self.yx[y] == -1:  # an exposed vertex in Y found, so augmenting path exists!
                            break
                        self.T[y] = True  # else just add y to T,
                        q[wr] = self.yx[y]  # add vertex yx[y], which is matched
                        wr += 1
                        self.__add_to_tree(self.yx[y], x)  # add edges (x,y) and (y,yx[y]) to the tree
                    y += 1
                if y < self.n:  # augmenting path found
                    break
            if y < self.n:
                break

            self.__update_labels()  # augmenting path not found, improve labeling
            wr = rd = 0
            y = 0
            steps = 0
            while y < self.n:
                # in this cycle we add edges that were added to the equality graph as a
                # result of improving the labeling, we add edge (slackx[y], y) to the tree if
                # and only if !T[y] && slack[y] == 0, also with this edge we add another one
                # (y, yx[y]) or augment the matching, if y was exposed
                if not self.T[y] and self.slack[y] == 0:
                    if self.yx[y] == -1:  # exposed vertex in Y found - augmenting path exists
                        x = self.slackx[y]
                        break
                    else:  # else just add y to T
                        self.T[y] = True
                        if not self.S[int(self.yx[y])]:
                            q[wr] = self.yx[y]  # add vertex yx[y], which is matched with
                        wr += 1  # y, to the queue
                        self.__add_to_tree(self.yx[y], self.slackx[y])  # and add edges (x,y) and (y,
                        # yx[y]) to the tree
                y += 1
            if y < self.n:
                break  # augmenting path found

        if y < self.n:  # we found an augmenting path
            self.max_match += 1
            # invert edges along the augmenting path
            cx = int(x)
            cy = int(y)
            while cx != -2:
                ty = self.xy[cx]
                self.yx[cy] = cx
                self.xy[cx] = cy
                cx = self.prev[cx]
                cy = ty
            self.__augment()

    def __call__(self, cost):
        if self.__reset(cost):
            self.__augment()
            ret = 0
            m = np.zeros((self.n, self.n), dtype=int)
            for i in range(self.n):
                ret += self.cost[i, self.xy[i]]
                m[i, self.xy[i]] = 1
            return True, m
        else:
            return False, None


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    # cost = np.array([
    #     [0.95, 0.76, 0.62, 0.41, 0.06],
    #     [0.23, 0.46, 0.79, 0.94, 0.35],
    #     [0.61, 0.02, 0.92, 0.92, 0.81],
    #     [0.49, 0.82, 0.74, 0.41, 0.01],
    #     [0.89, 0.44, 0.18, 0.89, 0.14]
    # ])
    times = []
    for i in range(5, 10):
        c = np.random.randn(i, i)
        hung = Hungarian()
        start = time.time()
        min_cost, matching = hung.solve(c)
        diff = time.time() - start
        print('Maximum assignment reward: {} in {} sec'.format(min_cost, diff))
        print('Maximal matching: \n{}'.format(matching))
        times.append(diff)
    #plt.plot(times)
    #plt.show()
