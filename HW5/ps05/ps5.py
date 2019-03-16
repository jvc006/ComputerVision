"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.matrix([[init_x, init_y, 0., 0.]]).T  # state
        self.covD = np.matrix(np.zeros((4, 4), dtype = np.float64))
        self.trasD = np.matrix([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.mT = np.matrix([[1., 0., 0., 0.],[0., 1., 0., 0.]])
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)

    def predict(self):
        self.state = self.trasD * self.state
        self.covD = self.trasD * self.covD * self.trasD.T + self.Q

    def correct(self, meas_x, meas_y):
        Kgain = self.covD * self.mT.T * np.linalg.inv(self.mT * self.covD * self.mT.T + self.R)
        Y_t = np.matrix([[meas_x, meas_y]]).T
        self.state = self.state + Kgain * (Y_t - self.mT * self.state)
        self.covD = self.covD - Kgain * self.mT * self.covD

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0, 0], self.state[1, 0]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        h, w, channel = self.frame.shape
        x_cod = np.random.choice(w, self.num_particles, True).astype(np.float32)
        y_cod = np.random.choice(h, self.num_particles, True).astype(np.float32)
        self.particles = np.stack((x_cod, y_cod), axis = -1)  # Initialize your particles array. Read the docstring.
        self.weights = np.array([1/self.num_particles] * self.num_particles)  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.
        self.index = np.arange(self.num_particles)

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """

        m, n = template.shape
        MSE = np.sum((1.*template - 1.*frame_cutout)**2.)/(m*n)
        res = np.exp(-(MSE)/(2. * self.sigma_exp**2))
        return res

        return NotImplementedError

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        j = np.random.choice(self.index, self.num_particles, True, p = self.weights)

        update_particles = self.particles[j]

        return update_particles

    def update_weight(self, frame):
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_grey = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        H, W = frame_grey.shape
        h, w = template_grey.shape

        loc_x = np.array((self.particles[:, 0])).astype(int)
        loc_y = np.array((self.particles[:, 1])).astype(int)
        loc_x = np.clip(loc_x, 0, W-1)
        loc_y = np.clip(loc_y, 0, H-1)

        bordersize = h//2 + 1
        Addborder=cv2.copyMakeBorder(frame_grey, top=bordersize, bottom=bordersize, \
             left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])

        loc_x = loc_x + bordersize
        loc_y = loc_y + bordersize

        top = loc_y + h//2
        if h%2 == 0 :
            bottom = loc_y - h//2
        else:
            bottom = loc_y - h//2 - 1

        right = loc_x + w//2
        if w%2 == 0 :
            left = loc_x - w//2
        else:
            left = loc_x -w//2 - 1

        frame_grey_Cuts = [Addborder[bottom[i]: top[i], left[i] : right[i]] \
                                        for i in range(self.num_particles)]

        self.weights = np.array([self.get_error_metric(template_grey, frame_grey_Cut) \
                                        for frame_grey_Cut in frame_grey_Cuts])
        self.weights /= np.sum(self.weights)

        Addbordercolor = cv2.copyMakeBorder(frame, top=bordersize, bottom=bordersize, \
             left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])       

        p = np.argmax(self.weights)
        bestCuts = Addbordercolor[bottom[p]: top[p], left[p] : right[p]]
        return bestCuts

    def calculate_weight(self, frame):
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_grey = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        H, W = frame_grey.shape
        h, w = template_grey.shape

        loc_x = np.array((self.particles[:, 0])).astype(int)
        loc_y = np.array((self.particles[:, 1])).astype(int)
        loc_x = np.clip(loc_x, 0, W-1)
        loc_y = np.clip(loc_y, 0, H-1)

        bordersize = h//2 + 1
        Addborder=cv2.copyMakeBorder(frame_grey, top=bordersize, bottom=bordersize, \
             left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])

        loc_x = loc_x + bordersize
        loc_y = loc_y + bordersize

        top = loc_y + h//2
        if h%2 == 0 :
            bottom = loc_y - h//2
        else:
            bottom = loc_y - h//2 - 1

        right = loc_x + w//2
        if w%2 == 0 :
            left = loc_x - w//2
        else:
            left = loc_x -w//2 - 1

        frame_grey_Cuts = [Addborder[bottom[i]: top[i], left[i] : right[i]] \
                                        for i in range(self.num_particles)]

        weights_temp = np.array([self.get_error_metric(template_grey, frame_grey_Cut) \
                                        for frame_grey_Cut in frame_grey_Cuts])
        weights_temp /= np.sum(weights_temp)
        return weights_temp

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
        bestcut = self.update_weight(frame)
        self.particles = self.resample_particles()


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            cv2.circle(frame_in, (int(self.particles[i,0]), int(self.particles[i,1])), 1, (0, 255, 0), -1)

        m, n, channel = self.template.shape
        cv2.rectangle(frame_in, (int(x_weighted_mean) - n // 2, int(y_weighted_mean) - m // 2),
                         (int(x_weighted_mean) + n // 2, int(y_weighted_mean) + m // 2), (0, 255, 0), 2)
        # Complete the rest of the code as instructed.

        weighted_mean_dis = 0
        for i in range(self.num_particles):
            dist = np.sqrt((self.particles[i, 0] - x_weighted_mean)**2 + \
                                    (self.particles[i, 1] - y_weighted_mean)**2)
            weighted_mean_dis += dist * self.weights[i]

        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(weighted_mean_dis), (250, 250, 250), 2)
        return frame_in


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
        bestcut = self.update_weight(frame)
        self.particles = self.resample_particles()
        self.template = (self.alpha * bestcut + (1 - self.alpha) * self.template).astype(np.uint8)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.initial_template = template
        self.count = 0
        self.std = 0
        self.calc_weights = 0.

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.count += 1

        # if self.count == 146:
        #     cv2.imshow('Tracking', frame)
        #     cv2.waitKey(0)


        weights_temp = self.calculate_weight(frame)
        # weights_temp = sorted(weights_temp, reverse=True)
        # self.calc_weights = np.sum(weights_temp[0:2])
        self.calc_weights = max(weights_temp)

        # print(self.calc_weights)

        if self.calc_weights > 0.01 and 200 > self.count > 50:
            # self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
            # cv2.imshow('Tracking', frame)
            # cv2.waitKey(0)
        
            pass

            # particles_temp = self.resample_particles()
    
            # self.particles = particles_temp

        else :
            self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
            bestcut = self.update_weight(frame)
            self.particles = self.resample_particles()


        ratio = .995**(self.count)
        self.template = cv2.resize(self.initial_template, (0,0), fx=ratio, fy=ratio) 
        # cv2.imshow('Tracking', self.template)
        # cv2.waitKey(1)



class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.initial_template = template
        self.count = 0
        self.std = 0
        self.calc_weights = 0.

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.count += 1

        # if self.count == 146:
        #     cv2.imshow('Tracking', frame)
        #     cv2.waitKey(0)


        weights_temp = self.calculate_weight(frame)
        # weights_temp = sorted(weights_temp, reverse=True)
        # self.calc_weights = np.sum(weights_temp[0:2])
        self.calc_weights = max(weights_temp)

        # print(self.calc_weights)

        if self.calc_weights > 0.01 and 200 > self.count > 50:
            # self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
            # cv2.imshow('Tracking', frame)
            # cv2.waitKey(0)
        
            pass

            # particles_temp = self.resample_particles()
    
            # self.particles = particles_temp

        else :
            self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
            bestcut = self.update_weight(frame)
            self.particles = self.resample_particles()


        ratio = .995**(self.count)
        self.template = cv2.resize(self.initial_template, (0,0), fx=ratio, fy=ratio) 
        # cv2.imshow('Tracking', self.template)
        # cv2.waitKey(1)

class MDParticleFilter_2(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter_2, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.initial_template = template
        self.count = 0
        self.std = 0
        self.calc_weights = 0.

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.count += 1

        # if self.count == 146:
        #     cv2.imshow('Tracking', frame)
        #     cv2.waitKey(0)


        weights_temp = self.calculate_weight(frame)
        # weights_temp = sorted(weights_temp, reverse=True)
        # self.calc_weights = np.sum(weights_temp[0:2])
        self.calc_weights = max(weights_temp)

        # print(self.calc_weights)

        if self.calc_weights > 0.002 and (20 > self.count > 12 or 42 > self.count > 34):
            # self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
            # cv2.imshow('Tracking', frame)
            # cv2.waitKey(0)
        
            pass

            # particles_temp = self.resample_particles()
    
            # self.particles = particles_temp

        else :
            self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
            bestcut = self.update_weight(frame)
            self.particles = self.resample_particles()


        ratio = .995**(self.count)
        self.template = cv2.resize(self.initial_template, (0,0), fx=ratio, fy=ratio) 
        # cv2.imshow('Tracking', self.template)
        # cv2.waitKey(1)

