import pandas as pd

def analyze():
    # Load population
    pop1 = pd.read_csv('data/logs/population_1.csv') # Rabbit
    pop8 = pd.read_csv('data/logs/population_8.csv') # Macaque
    
    print("Horned Rabbit (1) Population at start and end:")
    print(pop1[['Tick', 'Population']].iloc[[0, -1]])
    
    print("\nEmerald Macaque (8) Population at start and end:")
    print(pop8[['Tick', 'Population']].iloc[[0, -1]])
    
    # Load life events
    life = pd.read_csv('data/logs/life_events.csv')
    deaths_1 = life[(life['Species'] == 1) & (life['Event'] == 'death')]
    deaths_8 = life[(life['Species'] == 8) & (life['Event'] == 'death')]
    
    print("\nRabbit Deaths:", len(deaths_1))
    print("Macaque Deaths:", len(deaths_8))
    
    # Load combat logs to see who is killing them
    combat = pd.read_csv('data/logs/combat_log.csv')
    combat_deaths = combat[combat['Outcome'] == 'kill']
    
    rabbit_killed = combat_deaths[combat_deaths['Entity B'] == 1]['Entity A'].value_counts()
    print("\nSpecies that killed Rabbits:")
    print(rabbit_killed)
    
    macaque_killed = combat_deaths[combat_deaths['Entity B'] == 8]['Entity A'].value_counts()
    print("\nSpecies that killed Macaques:")
    print(macaque_killed)
    
    # What did the predators of rabbits eat mostly?
    if len(rabbit_killed) > 0:
        main_predator = rabbit_killed.index[0]
        print(f"\nMain predator of rabbits is {main_predator}")
        predator_kills = combat_deaths[combat_deaths['Entity A'] == main_predator]['Entity B'].value_counts()
        print(f"Species killed by {main_predator}:")
        print(predator_kills)

if __name__ == '__main__':
    analyze()
