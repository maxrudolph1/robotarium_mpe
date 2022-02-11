source ~/.bashrc


## Target Team

cp /home/mrudolph/Documents/multi_agent_learning/team_generation/assigned_teams.json \
/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json

cd /home/mrudolph/Documents/robotarium_experiments/sim_results
runs=20
echo assigned
python robotarium_results.py \
--task coverage \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_coverage/m_01000 \
--save_path  ./results/assigned \
--show False \
--runs $runs

cp /home/mrudolph/Documents/multi_agent_learning/team_generation/expert_teams.json \
/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json
cd /home/mrudolph/Documents/robotarium_experiments/sim_results

 #--policy_path ~/Documents/multi_agent_learning/expert/simple_coverage/checkpoint01000 \
echo expert
python robotarium_results.py \
--task coverage \
--policy_path ~/Documents/multi_agent_learning/expert/simple_coverage/checkpoint01000 \
--save_path  ./results/expert \
--show False \
--runs $runs

cp /home/mrudolph/Documents/multi_agent_learning/team_generation/uniform_teams.json \
/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json
cd /home/mrudolph/Documents/robotarium_experiments/sim_results

echo uniform
python robotarium_results.py \
--task coverage \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_coverage/m_01000 \
--save_path  ./results/uniform \
--show False \
--runs $runs

cd /home/mrudolph/Documents/robotarium_experiments/sim_results
cp /home/mrudolph/Documents/multi_agent_learning/team_generation/random_teams.json \
/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json

echo loc based
python robotarium_results.py \
--task coverage \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_coverage/m_01000 \
--save_path  ./results/loc_based \
--show False \
--runs $runs