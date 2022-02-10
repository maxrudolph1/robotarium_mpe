source ~/.bashrc

cd /home/mrudolph/Documents/robotarium_experiments/sim_results
cp /home/mrudolph/Documents/multi_agent_learning/team_generation/random_teams.json \
/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json

runs=400
echo loc based
python robotarium_results.py \
--task coverage \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_coverage/m_01000 \
--save_path  ./results/loc_based \
--show False \
--runs $runs


python robotarium_results.py \
--task navigation \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_navigation/m_01000 \
--save_path  ./results/loc_based \
--show False \
--runs $runs


python robotarium_results.py \
--task transport \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_transport/m_01000 \
--save_path  ./results/loc_based \
--show False \
--runs $runs

cd -