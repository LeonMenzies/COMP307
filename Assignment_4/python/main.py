from aem import con
from sqlalchemy import true
from sympy import total_degree
from utility import calculate_euclidean_distance
import utility as utility
import loader as loader
import numpy as np
import sys


def main():

    # Paths to the data and solution files.
    # vrp_file = "python/vrp-data/n80-k10.vrp"
    # sol_file = "python/vrp-data/n80-k10.sol"

    vrp_file = "python/vrp-data/n32-k5.vrp"
    sol_file = "python/vrp-data/n32-k5.sol"

    # Loading the VRP data file.
    px, py, demand, capacity, depot = loader.load_data(vrp_file)

    # Displaying to console the distance and visualizing the optimal VRP solution.
    # vrp_best_sol = loader.load_solution(sol_file)
    # best_distance = utility.calculate_total_distance(vrp_best_sol, px, py, depot)
    # print("Best VRP Distance:", best_distance)
    # utility.visualise_solution(vrp_best_sol, px, py, depot, "Optimal Solution")

    # Executing and visualizing the nearest neighbour VRP heuristic.
    # Uncomment it to do your assignment!

    # nnh_solution = nearest_neighbour_heuristic(px, py, demand, capacity, depot)
    # nnh_distance = utility.calculate_total_distance(nnh_solution, px, py, depot)
    # print("Nearest Neighbour VRP Heuristic Distance:", nnh_distance)
    # utility.visualise_solution(nnh_solution, px, py, depot, "Nearest Neighbour Heuristic")

    # Executing and visualizing the saving VRP heuristic.
    # Uncomment it to do your assignment!
    
    sh_solution = savings_heuristic(px, py, demand, capacity, depot)
    sh_distance = utility.calculate_total_distance(sh_solution, px, py, depot)
    print("Saving VRP Heuristic Distance:", sh_distance)
    utility.visualise_solution(sh_solution, px, py, depot, "Savings Heuristic")


def nearest_neighbour_heuristic(px, py, demand, capacity, depot):

    """
    Algorithm for the nearest neighbour heuristic to generate VRP solutions.

    :param px: List of X coordinates for each node.
    :param py: List of Y coordinates for each node.
    :param demand: List of each nodes demand.
    :param capacity: Vehicle carrying capacity.
    :param depot: Depot.
    :return: List of vehicle routes (tours).
    """

    routes = []
    visited = [depot]

    while len(visited) < len(px):

        route = [depot]
        total = 0
        current = depot

        while total <= capacity and len(visited) < len(px):

            nearest = -1
            distance = 9223372036854775807

            for i in range(len(px)):

                if visited.__contains__(i):
                    continue

                dis = calculate_euclidean_distance(px, py, current, i)
                temp_total = demand[i] + total

                if dis < distance and temp_total <= capacity:
                    nearest = i
                    distance = dis

            current = nearest
            if current == -1:
                break     

            route.append(nearest)
            total += demand[nearest]
            visited.append(nearest)
           
        route.append(depot)
        routes.append(route)

    return routes


def savings_heuristic(px, py, demand, capacity, depot):

    """
    Algorithm for Implementing the savings heuristic to generate VRP solutions.

    :param px: List of X coordinates for each node.
    :param py: List of Y coordinates for each node.
    :param demand: List of each nodes demand.
    :param capacity: Vehicle carrying capacity.
    :param depot: Depot.
    :return: List of vehicle routes (tours).
    """

    #Initilise routes except depot
    routes = {}

    for i in range(len(px)):
        if i == depot:
                continue
        routes[i] = (depot, i, depot)


    #Compute the savings table
    saving_table = {}
    for i in range(len(px)):
        if i == depot:
                continue
        for j in range(len(px)):
            if j == depot or j == i:
                continue
            
            saving_table[(i,j)] = calculate_euclidean_distance(px, py, i, depot) + calculate_euclidean_distance(px, py, depot, j) - calculate_euclidean_distance(px, py, i, j)  

    #Sort by value
    saving_table = dict(sorted(saving_table.items(), key=lambda item: item[1]))

    visited = []
    running = True
    while running:

        #Check for best merge
        for val in saving_table:

            if visited.__contains__(val):
                continue
            
            route1 = routes[val[0]]
            route2 = routes[val[1]]

            #Create the temp route
            new_route = [depot]
            for i in route1:
                if i != depot:
                    new_route.append(i)

            for i in route2:
                if i != depot:
                    new_route.append(i)
            new_route.append(depot)

            #Check if the route is feasable
            if check_route(demand, new_route, capacity):
                
                del routes[val[1]]
                routes[val[0]] = new_route
                # routes[val[1]] = new_route


                #Remove from saving table
                visited.append(val)
                for val1 in saving_table:
                    if val1[0] == val[0]:
                        visited.append(val1)
                    elif val1[1] == val[1]:
                        visited.append(val1)
                    elif val1[1] == val[0] and val1[0] == val[1]:
                        visited.append(val1)

            continue
        
        running = False

    return list(routes.values())


def check_route(demand, route, capacity):
    total = 0
    for i in route:
        total += demand[i]

    if total <= capacity:
        return True
    else:
        return False

if __name__ == '__main__':
    main()
