from matplotlib.style import available
import numpy as np
import copy
import multiprocessing
import torch
import random



class GAevolver:
    def __init__(self, population_size, generations):
        self.available_jobs = [] # [[gen_idx, model], ] --- Queue of jobs
        self.available_jobs_mutex = multiprocessing.Lock()
        self.finished_jobs = [] # [[gen_idx, model, reward], ] # Ordered list with the indexes essentially a copy of available_jobs, but order not changing, so model doesnt need to be sent back
        self.finished_jobs_mutex = multiprocessing.Lock()
        self.population_size = population_size
        self.generations = generations
        self.completed_jobs = 0
        self.operating = False # Bool is True if operations is being performed, aka no jobs should be popped here
        self.model_input = 0
        self.model_hidden = []
        self.model_output = 0
        self.elitism = -1
        self.top_best_actors_mutates = -1
        self.random_mutation_percent = -1
        self.amount_of_nonbest_actors_mutates = -1
        self.best_model = None
        self.best_generation_reward = 0

    def set_model_parameters(self, input, hidden, output):
        self.model_input = input
        self.model_output = output
        self.model_hidden = hidden

    def set_evolution_parameters(self, elitism=-1, top_best_actors_mutates=-1, random_mutation_percent=20, amount_of_nonbest_actors_mutates=-1):
        self.elitism = elitism
        self.top_best_actors_mutates = top_best_actors_mutates
        self.random_mutation_percent = random_mutation_percent
        self.amount_of_nonbest_actors_mutates = amount_of_nonbest_actors_mutates


    def generate_initial_jobs(self):
        #print("TEST: Generating initial jobs")
        self.available_jobs_mutex.acquire()
        self.available_jobs = []
        for i in range(self.population_size):
            self.available_jobs.append([i, self.make_model(self.model_input, self.model_hidden, self.model_output)])
        self.available_jobs_mutex.release()

        # A deep copy (so all list of lists is copied) should be stored in finished jobs append the reward to self.finished_jobs[gen_idx].append(REWARD)
        self.finished_jobs_mutex.acquire()
        self.finished_jobs = copy.deepcopy(self.available_jobs)
        self.finished_jobs_mutex.release()

    # Function to call each generaton (to mutate and generate new jobs)
    def on_generation(self):
        self.finished_jobs_mutex.acquire()
        self.available_jobs_mutex.acquire()

        # Re generating self.available_jobs list
        self.available_jobs = []
        for i in range(self.population_size):
            self.available_jobs.append([i, None])

        if len(self.finished_jobs) != self.population_size:
            print("Error: There are still remaining available jobs")
            print("len(self.finished_jobs) = ", len(self.finished_jobs))
            print("len(self.available_jobs) = ", len(self.available_jobs))

        # Sort the jobs so they are sorted by rewards
        def sort_jobs(job):
            return job[2]
        self.finished_jobs.sort(reverse=True, key=sort_jobs)
        self.best_model = self.finished_jobs[0][1]
        self.best_generation_reward = self.finished_jobs[0][2]

        # Create new jobs using the evolution parameters
        for i in range(0, self.elitism):
            self.available_jobs[i][1] = copy.deepcopy(self.finished_jobs[i][1]) # Update the model

        best_mutation_amount = self.population_size - self.amount_of_nonbest_actors_mutates
        for i in range(self.elitism, best_mutation_amount):
            rand_best_int = random.randint(0, self.top_best_actors_mutates - 1)
            self.available_jobs[i][1] = copy.deepcopy(self.finished_jobs[rand_best_int][1]) # Update the model
        
        for i in range(best_mutation_amount, self.population_size):
            rand_worst_int = random.randint(self.top_best_actors_mutates, self.population_size - 1)
            self.available_jobs[i][1] = copy.deepcopy(self.finished_jobs[rand_worst_int][1]) # Update the model

        # Mutate the jobs
        for i in range(len(self.available_jobs)):
            self.mutate_model_random(model=self.available_jobs[i][1], mutation_rate_percent=self.random_mutation_percent)


        self.finished_jobs = copy.deepcopy(self.available_jobs)

        self.finished_jobs_mutex.release()
        self.available_jobs_mutex.release()
        return 0

    def append_job(self, job):
        self.available_jobs_mutex.acquire()
        self.available_jobs.append(job)
        self.available_jobs_mutex.release()

    def add_finished_job(self, job_idx, reward):
        self.finished_jobs_mutex.acquire()  # Should not be necessary since the position accessed should never be the same, but this is just a safety feature, may slow down the program though
        self.finished_jobs[job_idx].append(reward)
        self.completed_jobs += 1 # Protected by finished_jobs_mutex
        self.finished_jobs_mutex.release()

    def init_weights_normal_dist(self, m):    # Initialize weights from 0 to 1 with a uniform distribution
        if type(m) == torch.nn.Linear:
            torch.nn.init.uniform_(m.weight, a=-1.0, b=1.0)
            torch.nn.init.uniform_(m.bias.data, a=-1.0, b=1.0)
            #m.bias.data.fill_(0.01)

    # Define the GA Neural Network Using pygads' pytorch api
    def make_model(self, input, hidden, output, activation=torch.nn.ReLU()):
        # example input = 12, 3 hidden layers of 30, output = 3 --> 12, [30, 30, 30], 3 --> 12-30, 30-30, 30,30, 30-3 --> Loop should be (hidden + 1) iterations
        list_seq = [torch.nn.Linear(input, hidden[0]), torch.nn.ReLU()] # Initial layer

        for i in range(len(hidden) - 1): # Hidden Layers
            list_seq.append(torch.nn.Linear(hidden[i], hidden[i+1]))
            list_seq.append(torch.nn.ReLU())

        list_seq.append(torch.nn.Linear(hidden[len(hidden) - 1], output))
        list_seq.append(torch.nn.ReLU())

        model = torch.nn.Sequential(*list_seq) # Need to unpack the list to use it in sequential
        model.apply(self.init_weights_normal_dist)
        return model

    def mutate_model_random(self, model, mutation_rate_percent):
        for name, param in model.named_parameters():
            shape = param.shape

            mask = np.random.choice([1, 0], size=shape, p=[mutation_rate_percent/100, (100-mutation_rate_percent)/100])
            inv_mask = (mask-1)*-1

            rands = torch.randn(shape)
            param.data = ((rands * mask) + (param.cpu().detach().numpy() * inv_mask)).float()
    






