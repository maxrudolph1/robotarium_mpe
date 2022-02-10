source ~/.bashrc

cd /home/mrudolph/Documents/robotarium_experiments/sim_results
cp /home/mrudolph/Documents/multi_agent_learning/team_generation/random_teams.json \
/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json

runs=400
echo combined
python robotarium_results.py \
--task combined \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_combined/m_01000 \
--save_path  ./results/combined/ \
--show False \
--runs $runs

cd -