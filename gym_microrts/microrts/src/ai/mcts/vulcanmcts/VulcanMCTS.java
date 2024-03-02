/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ai.mcts.vulcanmcts;

import ai.*;
import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import ai.evaluation.EvaluationFunction;
import ai.evaluation.SimpleSqrtEvaluationFunction3;
import ai.rewardfunction.RewardFunctionInterface;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import rts.GameState;
import rts.PlayerAction;
import rts.TraceEntry;
import rts.units.UnitTypeTable;
import ai.core.InterruptibleAI;
import java.lang.Math;
import java.lang.reflect.Array;

/**
 *
 * @author gustavo
 */
public class VulcanMCTS extends AIWithComputationBudget implements InterruptibleAI {
    public static int DEBUG = 0;
    public EvaluationFunction ef = null;
       
    Random r = new Random();
    public AI playoutPolicy = new RandomBiasedAI();

    public AI[] ais;

    public long max_actions_so_far = 0;
    
    public GameState gs_to_start_from = null;
    public VulcanMCTSNode tree = null;
    public int current_iteration = 0;
            
    public int MAXSIMULATIONTIME = 2048; // 1024
    public int MAX_TREE_DEPTH = 100;
    
    public int player;
    
    public float epsilon_0 = 0.2f;
    public float epsilon_l = 0.25f;
    public float epsilon_g = 0.0f;

    // these variables are for using a discount factor on the epsilon values above. 
    // My experiments indicate that things work better without discount
    // So, they are just maintained here for completeness:
    public float initial_epsilon_0 = 0.2f;
    public float initial_epsilon_l = 0.25f;
    public float initial_epsilon_g = 0.0f;
    public float discount_0 = 0.999f;
    public float discount_l = 0.999f;
    public float discount_g = 0.999f;
    
    public int global_strategy = VulcanMCTSNode.E_GREEDY;
    public boolean forceExplorationOfNonSampledActions = true;
    
    // statistics:
    public long total_runs = 0;
    public long total_cycles_executed = 0;
    public long total_actions_issued = 0;
    public long total_time = 0;
    


    //Vulcan
    public static final int RBF_EVAL_BASED = 0;
    public static final int RBF_REWARDS_BASED = 1;
    public int selected_rbf = RBF_EVAL_BASED;

    public float rbf_delta = 0.01f;
    public float rbf_epsilon = 1.0f;
    public int ser_n_actions = 5;
    public int ser_factor = 10;
    public double[] reward_weights = {10.0, 1.0, 1.0, 0.2, 1.0, 4.0};

    public ArrayList<Double> global_evals;
    public ArrayList<Double> global_risks;
    public double global_ser;
    public double global_rbf;
    public double global_last_eval;
    public ArrayList<Double> global_scores_0;
    public ArrayList<Double> global_scores_1;
    public ArrayList<ArrayList<Double>> global_rewards;
    public double global_final_reward;

    public int count = 0;

    public ArrayList<Integer> fallback_actions = new ArrayList<>();
    
    public RewardFunctionInterface[] rfs;

    // storage
    double[] rewards;
    boolean[] dones;
    PlayerAction pa1;
    PlayerAction pa2;
    

    public VulcanMCTS(UnitTypeTable utt) {
        this(100,-1,100,10,
             0.3f, 0.0f, 0.4f,
             new RandomBiasedAI(),
             new SimpleSqrtEvaluationFunction3(), true);
        startAIs();
    }    
    public VulcanMCTS(int available_time, int max_playouts, int lookahead, int max_depth, 
                               float e_l, float discout_l,
                               float e_g, float discout_g, 
                               float e_0, float discout_0, 
                               AI policy, EvaluationFunction a_ef,
                               boolean fensa) {
        super(available_time, max_playouts);
        MAXSIMULATIONTIME = lookahead;
        playoutPolicy = policy;
        MAX_TREE_DEPTH = max_depth;
        initial_epsilon_l = epsilon_l = e_l;
        initial_epsilon_g = epsilon_g = e_g;
        initial_epsilon_0 = epsilon_0 = e_0;
        discount_l = discout_l;
        discount_g = discout_g;
        discount_0 = discout_0;
        ef = a_ef;
        forceExplorationOfNonSampledActions = fensa;
        startAIs();
    }    
    public VulcanMCTS(int available_time, int max_playouts, int lookahead, int max_depth, float e_l, float e_g, float e_0, AI policy, EvaluationFunction a_ef, boolean fensa) {
        super(available_time, max_playouts);
        MAXSIMULATIONTIME = lookahead;
        playoutPolicy = policy;
        MAX_TREE_DEPTH = max_depth;
        initial_epsilon_l = epsilon_l = e_l;
        initial_epsilon_g = epsilon_g = e_g;
        initial_epsilon_0 = epsilon_0 = e_0;
        discount_l = 1.0f;
        discount_g = 1.0f;
        discount_0 = 1.0f;
        ef = a_ef;
        forceExplorationOfNonSampledActions = fensa;
        startAIs();
    }    
    public VulcanMCTS(int available_time, int max_playouts, int lookahead, int max_depth, float e_l, float e_g, float e_0, int a_global_strategy, AI policy, EvaluationFunction a_ef, boolean fensa) {
        super(available_time, max_playouts);
        MAXSIMULATIONTIME = lookahead;
        playoutPolicy = policy;
        MAX_TREE_DEPTH = max_depth;
        initial_epsilon_l = epsilon_l = e_l;
        initial_epsilon_g = epsilon_g = e_g;
        initial_epsilon_0 = epsilon_0 = e_0;
        discount_l = 1.0f;
        discount_g = 1.0f;
        discount_0 = 1.0f;
        global_strategy = a_global_strategy;
        ef = a_ef;
        forceExplorationOfNonSampledActions = fensa;
        startAIs();
    }        
    public VulcanMCTSNode getTree() {
        return tree;
    }
    public GameState getGameStateToStartFrom() {
        return gs_to_start_from;
    }
    @Override
    public String toString() {
        return getClass().getSimpleName() + "(" + TIME_BUDGET + ", " + ITERATIONS_BUDGET + ", " + MAXSIMULATIONTIME + "," + MAX_TREE_DEPTH + "," + epsilon_l + ", " + discount_l + ", " + epsilon_g + ", " + discount_g + ", " + epsilon_0 + ", " + discount_0 + ", " + playoutPolicy + ", " + ef + ")";
    }
    @Override
    public String statisticsString() {
        return "Total runs: " + total_runs + 
               ", runs per action: " + (total_runs/(float)total_actions_issued) + 
               ", runs per cycle: " + (total_runs/(float)total_cycles_executed) + 
               ", average time per cycle: " + (total_time/(float)total_cycles_executed) + 
               ", max branching factor: " + max_actions_so_far;
    }
    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        
        parameters.add(new ParameterSpecification("TimeBudget",int.class,100));
        parameters.add(new ParameterSpecification("IterationsBudget",int.class,-1));
        parameters.add(new ParameterSpecification("PlayoutLookahead",int.class,100));
        parameters.add(new ParameterSpecification("MaxTreeDepth",int.class,10));
        
        parameters.add(new ParameterSpecification("E_l",float.class,0.3));
        parameters.add(new ParameterSpecification("Discount_l",float.class,1.0));
        parameters.add(new ParameterSpecification("E_g",float.class,0.0));
        parameters.add(new ParameterSpecification("Discount_g",float.class,1.0));
        parameters.add(new ParameterSpecification("E_0",float.class,0.4));
        parameters.add(new ParameterSpecification("Discount_0",float.class,1.0));
                
        parameters.add(new ParameterSpecification("DefaultPolicy",AI.class, playoutPolicy));
        parameters.add(new ParameterSpecification("EvaluationFunction", EvaluationFunction.class, new SimpleSqrtEvaluationFunction3()));

        parameters.add(new ParameterSpecification("ForceExplorationOfNonSampledActions",boolean.class,true));
        
        return parameters;
    }    
    public int getPlayoutLookahead() {
        return MAXSIMULATIONTIME;
    }
    public void setPlayoutLookahead(int a_pola) {
        MAXSIMULATIONTIME = a_pola;
    }
    public int getMaxTreeDepth() {
        return MAX_TREE_DEPTH;
    }
    public void setMaxTreeDepth(int a_mtd) {
        MAX_TREE_DEPTH = a_mtd;
    }
    public float getE_l() {
        return epsilon_l;
    }
    public void setE_l(float a_e_l) {
        epsilon_l = a_e_l;
    }
    public float getDiscount_l() {
        return discount_l;
    }
    public void setDiscount_l(float a_discount_l) {
        discount_l = a_discount_l;
    }
    public float getE_g() {
        return epsilon_g;
    }
    public void setE_g(float a_e_g) {
        epsilon_g = a_e_g;
    }
    public float getDiscount_g() {
        return discount_g;
    }
    public void setDiscount_g(float a_discount_g) {
        discount_g = a_discount_g;
    }
    public float getE_0() {
        return epsilon_0;
    }
    public void setE_0(float a_e_0) {
        epsilon_0 = a_e_0;
    }
    public float getDiscount_0() {
        return discount_0;
    }
    public void setDiscount_0(float a_discount_0) {
        discount_0 = a_discount_0;
    }
    public AI getDefaultPolicy() {
        return playoutPolicy;
    }
    public void setDefaultPolicy(AI a_dp) {
        playoutPolicy = a_dp;
    }
    public EvaluationFunction getEvaluationFunction() {
        return ef;
    }
    public void setEvaluationFunction(EvaluationFunction a_ef) {
        ef = a_ef;
    }
    public boolean getForceExplorationOfNonSampledActions() {
        return forceExplorationOfNonSampledActions;
    }
    public void setForceExplorationOfNonSampledActions(boolean fensa) {
        forceExplorationOfNonSampledActions = fensa;
    }    
    public void setRewardFunctions(RewardFunctionInterface[] rfs) {
        this.rfs = rfs;
    } 
    public void setRBFDelta(float rbf_delta) {
        this.rbf_delta = rbf_delta;
    }
    public void setRBFEpsilon(float rbf_epsilon) {
        this.rbf_epsilon = rbf_epsilon;
    }
    public void setSERNActions(int ser_n_actions) {
        this.ser_n_actions = ser_n_actions;
    }
    public void setSERFactor(int ser_factor) {
        this.ser_factor = ser_factor;
    }
    public void setRewardWeights(double[] reward_weights) {
        this.reward_weights = reward_weights;
    }
    public void setSelectedRBF(int selected_rbf) {
        this.selected_rbf = selected_rbf;
    }

    public void reset() {
        tree = null;
        gs_to_start_from = null;
        total_runs = 0;
        total_cycles_executed = 0;
        total_actions_issued = 0;
        total_time = 0;
        current_iteration = 0;
    }    
        
    public AI clone() {
        return new VulcanMCTS(TIME_BUDGET, ITERATIONS_BUDGET, MAXSIMULATIONTIME, MAX_TREE_DEPTH, epsilon_l, discount_l, epsilon_g, discount_g, epsilon_0, discount_0, playoutPolicy, ef, forceExplorationOfNonSampledActions);
    }    
    
    private void startAIs(){
        int ITERATIONS_BUDGET = 10;
        ais = new AI[5];
        ais[0] = new PassiveAI();
        ais[1] = new RandomAI();
        ais[2] = new RandomBiasedAI();
        ais[3] = new RandomBiasedSingleUnitAI();
        ais[4] = new RandomNoAttackAI(ITERATIONS_BUDGET);
    }
    
    public AI selectRandomAi() {
        int rnd = new Random().nextInt(ais.length);
        return ais[rnd];
    }

    public PlayerAction getAction(int player, GameState gs) throws Exception
    {
        if (gs.canExecuteAnyAction(player)) {
            startNewComputation(player,gs.clone());
            computeDuringOneGameFrame();

            PlayerAction action = getBestActionSoFar();

            if (action.isEmpty()) {
                if (DEBUG>=1) System.out.println("VulcanMCTS: the selected action was empty! (returning a random action)");
                AI selectedAi = selectRandomAi();
                action = selectedAi.getAction(player, gs);
                fallback_actions.add(1);
            }
            else{
                fallback_actions.add(0);
            }
            return action;

        } else {
            return new PlayerAction();        
        }       
    }
    
    public void startNewComputation(int a_player, GameState gs) throws Exception {
        player = a_player;
        current_iteration = 0;
        tree = new VulcanMCTSNode(player, 1-player, gs, null, ef.upperBound(gs), current_iteration++, forceExplorationOfNonSampledActions);
        
        if (tree.moveGenerator==null) {
            max_actions_so_far = 0;
        } else {
            max_actions_so_far = Math.max(tree.moveGenerator.getSize(),max_actions_so_far);        
        }
        gs_to_start_from = gs;
        
        epsilon_l = initial_epsilon_l;
        epsilon_g = initial_epsilon_g;
        epsilon_0 = initial_epsilon_0;        
    }    
    
    public void resetSearch() {
        if (DEBUG>=2) System.out.println("Resetting search...");
        tree = null;
        gs_to_start_from = null;
    }

    public void computeDuringOneGameFrame() throws Exception {        
        if (DEBUG>=2) System.out.println("Search...");
        long start = System.currentTimeMillis();
        long end = start;
        long count = 0;
        while(true) {
            if (!iteration(player)) break;
            count++;
            end = System.currentTimeMillis();
            if (TIME_BUDGET>=0 && (end - start)>=TIME_BUDGET) break; 
            if (ITERATIONS_BUDGET>=0 && count>=ITERATIONS_BUDGET) break;             
        }
//        System.out.println("HL: " + count + " time: " + (System.currentTimeMillis() - start) + " (" + available_time + "," + max_playouts + ")");
        total_time += (end - start);
        total_cycles_executed++;
    }

    public PlayerAction getBestActionSoFar() {
        int idx = getMostVisitedActionIdx();
        //int idx = getHighestEvaluationActionIdx();
        if (idx==-1) {
            if (DEBUG>=1) System.out.println("VulcanMCTS no children selected. Returning an empty action");
            return new PlayerAction();
        }
        if (DEBUG>=2) tree.showNode(0,1,ef);
        if (DEBUG>=1) {
            VulcanMCTSNode best = (VulcanMCTSNode) tree.children.get(idx);
            System.out.println("VulcanMCTS selected children " + tree.actions.get(idx) + " explored " + best.visit_count + " Avg evaluation: " + (best.accum_evaluation/((double)best.visit_count)));
        }
        return tree.actions.get(idx);
    }
    
    public int getMostVisitedActionIdx() {
        total_actions_issued++;
            
        int bestIdx = -1;
        VulcanMCTSNode best = null;
        if (DEBUG>=2) {
//            for(Player p:gs_to_start_from.getPlayers()) {
//                System.out.println("Resources P" + p.getID() + ": " + p.getResources());
//            }
            System.out.println("Number of playouts: " + tree.visit_count);
            tree.printUnitActionTable();
        }
        if (tree.children==null) return -1;
        for(int i = 0;i<tree.children.size();i++) {
            VulcanMCTSNode child = (VulcanMCTSNode)tree.children.get(i);
            if (DEBUG>=2) {
                System.out.println("child " + tree.actions.get(i) + " explored " + child.visit_count + " Avg evaluation: " + (child.accum_evaluation/((double)child.visit_count)));
            }
//            if (best == null || (child.accum_evaluation/child.visit_count)>(best.accum_evaluation/best.visit_count)) {
            if (best == null || child.visit_count>best.visit_count) {
                best = child;
                bestIdx = i;
            }
        }
        
        return bestIdx;
    }
    
    public int getHighestEvaluationActionIdx() {
        total_actions_issued++;
            
        int bestIdx = -1;
        VulcanMCTSNode best = null;
        if (DEBUG>=2) {
//            for(Player p:gs_to_start_from.getPlayers()) {
//                System.out.println("Resources P" + p.getID() + ": " + p.getResources());
//            }
            System.out.println("Number of playouts: " + tree.visit_count);
            tree.printUnitActionTable();
        }
        for(int i = 0;i<tree.children.size();i++) {
            VulcanMCTSNode child = (VulcanMCTSNode)tree.children.get(i);
            if (DEBUG>=2) {
                System.out.println("child " + tree.actions.get(i) + " explored " + child.visit_count + " Avg evaluation: " + (child.accum_evaluation/((double)child.visit_count)));
            }
//            if (best == null || (child.accum_evaluation/child.visit_count)>(best.accum_evaluation/best.visit_count)) {
            if (best == null || (child.accum_evaluation/((double)child.visit_count))>(best.accum_evaluation/((double)best.visit_count))) {
                best = child;
                bestIdx = i;
            }
        }
        
        return bestIdx;
    }
      
    public void simulate(GameState gs, int time) throws Exception {
        boolean gameover = false;

        do{
            if (gs.isComplete()) {
                gameover = gs.cycle();
            } else {
                gs.issue(playoutPolicy.getAction(0, gs));
                gs.issue(playoutPolicy.getAction(1, gs));
            }
        }while(!gameover && gs.getTime()<time);   
    }

    

    // Vulcan

    public double sequence_execution_risk(ArrayList<Double> evals){
        double prod_safety = 1;
        for (int i = evals.size() - 1; i >= Math.max(evals.size() - ser_n_actions, 0); i--){
            double risk = 0.0f;
            double local_evaluation = evals.get(i);
            if (local_evaluation >= 0){ 
                // not a risky state
                risk = 0.0f;
            }
            else{ 
                // taking risks between [0,1]
                risk = Math.abs(local_evaluation);
            }

            prod_safety = prod_safety * (1 - (risk / ser_factor));
        }
        //                 risk        / safety
        double ser = (1 - prod_safety) / prod_safety;
        return ser;
    }
      
    public double risk_bounding_function_evalbased(ArrayList<Double> evals){

        // sufficient local conditions
        double slc = 0;
        for (int i = 0; i < evals.size(); i++){
            slc = slc + (Math.pow(0.99, i) * (evals.get(i)));
        }
    
        double rbf = rbf_epsilon + rbf_delta * slc; 

        return rbf;
    }

    public double risk_bounding_function_rewardsbased(double final_reward){
        double rbf = rbf_epsilon + rbf_delta * final_reward;
        return rbf;
    }

    public double calculate_reward(ArrayList<ArrayList<Double>> rewards){
        double final_reward = 0;
        for (int j = 0; j < rewards.size(); j++){
            ArrayList<Double> element_reward = rewards.get(j);
            double weight = reward_weights[j];
            for (int i = 0; i < element_reward.size(); i++){
                final_reward = final_reward + (Math.pow(0.99, i) * element_reward.get(i) * weight);
            }
        }
        return final_reward;
    }

    public double risk_bounding_function(ArrayList<Double> evals, ArrayList<ArrayList<Double>> tmp_rewards){

        global_final_reward = calculate_reward(tmp_rewards);

        double rbf = 0;
        if (selected_rbf == RBF_EVAL_BASED){
            rbf = risk_bounding_function_evalbased(evals);
        }
        else if (selected_rbf == RBF_REWARDS_BASED){
            rbf = risk_bounding_function_rewardsbased(global_final_reward);
        }
        return rbf;
    }
    
    
    public boolean iteration(int player) throws Exception {
        VulcanMCTSNode leaf = tree.selectLeaf(player, 1-player, epsilon_l, epsilon_g, epsilon_0, global_strategy, MAX_TREE_DEPTH, current_iteration++);

        if (leaf!=null) {
            count++;

            GameState gs2 = leaf.gs.clone();
            ArrayList<ArrayList<Double>> response = simulate_evaluated(gs2, gs2.getTime() + MAXSIMULATIONTIME, player);

            ArrayList<Double> evals = response.get(0);
            ArrayList<Double> risks = response.get(1);
            global_scores_0 = response.get(2);  // self
            global_scores_1 = response.get(3);  // enemy
            
            global_evals = evals;
            global_risks = risks;

            // Rewards
            ArrayList<ArrayList<Double>> tmp_rewards = new ArrayList<>();
            for (int i = 4; i < response.size(); i++) {
                tmp_rewards.add(response.get(i));
            }
            global_rewards = tmp_rewards;
            
            double local_evaluation = ef.evaluate(player, 1-player, gs2);
            global_last_eval = local_evaluation;
            
            // evaluation
            int time = gs2.getTime() - gs_to_start_from.getTime();
            double evaluation = local_evaluation * Math.pow(0.99, time/10.0);
            
            // SER
            double ser = sequence_execution_risk(evals);
            global_ser = ser;
            
            // RBF
            double rbf = risk_bounding_function(evals, tmp_rewards);
            global_rbf = rbf;

            if (ser <= rbf){
            //if ((ser <= rbf) && (ser >= 0.4 * rbf)){
                // State history satisfies the bounding function

                // update the node
                // TODO: comparar com o pseudocodigo
                //leaf.propagateEvaluation(evaluation,null);            
                leaf.propagateEvaluation(rbf-1, null);
                leaf.setRisk(ser);

                // update the epsilon values:
                epsilon_0*=discount_0;
                epsilon_l*=discount_l;
                epsilon_g*=discount_g;
                total_runs++;
            
            }
            else{
                if (DEBUG>=2) System.out.println("violates bounding function, removing node from tree...");
                // State history violates the bounding function
                // delete child
                leaf.parent.children.remove(leaf);
                leaf.parent = null;
                leaf = null;
                return false;
            }
            
        } else {
            // no actions to choose from :)

            System.err.println(this.getClass().getSimpleName() + ": claims there are no more leafs to explore...");
            return false;
        }
        return true;
    }


    public ArrayList<ArrayList<Double>> simulate_evaluated(GameState gs, int time, int player) throws Exception {
        if (DEBUG>=2) System.out.println("simulate_evaluated...");
        boolean gameover = false;
        ArrayList<Double> evals = new ArrayList<>();
        ArrayList<Double> risks = new ArrayList<>();
        ArrayList<Double> scores_0 = new ArrayList<>();
        ArrayList<Double> scores_1 = new ArrayList<>();

        ArrayList<Double> rewards0 = new ArrayList<>();
        ArrayList<Double> rewards1 = new ArrayList<>();
        ArrayList<Double> rewards2 = new ArrayList<>();
        ArrayList<Double> rewards3 = new ArrayList<>();
        ArrayList<Double> rewards4 = new ArrayList<>();
        ArrayList<Double> rewards5 = new ArrayList<>();

        rewards = new double[rfs.length];
        dones = new boolean[rfs.length];

        do{
            if (gs.isComplete()) {
                gameover = gs.cycle();
            } else {
                // Select random ai
                AI selectedAi = selectRandomAi();

                // Issue action for both players
                pa1 = selectedAi.getAction(0, gs);
                pa2 = selectedAi.getAction(1, gs);
                gs.issue(pa1);
                gs.issue(pa2);
                TraceEntry te  = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
                te.addPlayerAction(pa1.clone());
                te.addPlayerAction(pa2.clone());

                // Evaluate state E()
                double local_evaluation = ef.evaluate(player, 1-player, gs);
                evals.add(local_evaluation);
                
                // Evaluate scores
                double score_0 = ef.base_score(player, gs);
                double score_1 = ef.base_score(1-player, gs);
                scores_0.add(score_0);
                scores_1.add(score_1);

                // Evaluate risk
                double risk = 0.0f;
                // player maximiza
                if (local_evaluation >= 0){ // nao corre risco
                    risk = 0.0f;
                }
                else{ // corre risco entre [0,1]
                    risk = Math.abs(local_evaluation);
                }
                risks.add(risk);

                // Evaluate rewards
                for (int i = 0; i < rewards.length; i++) {
                    rfs[i].computeReward(player, 1 - player, te, gs);
                    dones[i] = rfs[i].isDone();
                    rewards[i] = rfs[i].getReward();
                }
                rewards0.add(rewards[0]);
                rewards1.add(rewards[1]);
                rewards2.add(rewards[2]);
                rewards3.add(rewards[3]);
                rewards4.add(rewards[4]);
                rewards5.add(rewards[5]);

            }
        }while(!gameover && gs.getTime()<time);
        if (DEBUG>=2) System.out.println("finished simulation...");

        
        // return evals and risks
        ArrayList<ArrayList<Double>> response = new ArrayList<>();
        response.add(evals);
        response.add(risks);
        response.add(scores_0);
        response.add(scores_1);
        response.add(rewards0);
        response.add(rewards1);
        response.add(rewards2);
        response.add(rewards3);
        response.add(rewards4);
        response.add(rewards5);

        return response;
    }


}
