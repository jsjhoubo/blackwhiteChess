第八关帮我做所有测试， 还有训练胜率一类的
// Online C++ compiler to run C++ program online
#include <iostream>
#include <vector>
#include <utility>
#include <array>
#include <random>
#include <ctime>

const int BLACK = 1;
const int WHITE = -1;
const int EMPTY = 0;

struct Board {
    Board(){ init_start(); }
    int8_t a[64];               // row-major
    int8_t at(int r, int c) const {
        return a[r*8 +c];
    };
    int8_t& at(int r, int c) {
        return a[r*8 +c];
    }

    void clear() {
        for (int i=0;i<64;i++) {
            a[i] =0;
        }
    };               // 全置 0
    void init_start() {
        clear();
        at(3,3) =-1;
        at(3,4) =1;
        at(4,3) =1;
        at(4,4) =-1;
    };          
    void print() {
        for(int i=0;i<8;i++) {
            for (int j=0;j<8;j++) {
                if (at(i,j) ==0) {
                    printf(".");
                }
                else if (at(i,j) ==1) {
                    printf("B");
                }
                else printf("W");
            }
            printf("\n");
        }
    }
    bool in_range(int r,  int c) const {
        if (r>=0 && r<8 && c>=0 && c<8) {
            return true;
        }
        return false;
    }
};
static const int dr[8] = {-1,-1,-1,0,0,1,1,1};
static const int dc[8] = {-1,0,1,-1,1,-1,0,1};

int flips_if_place(const Board& b, int player, int r, int c) {
    if (b.in_range(r, c)== false || b.at(r, c) !=EMPTY) {
        return 0;
    }
    int cnt =0;
    for (int i=0;i<8;i++) {
        int r1 =r + dr[i];
        int c1 =c + dc[i];
        int tmp =0;
        while (b.in_range(r1, c1) && b.at(r1, c1) == -player) {
            ++tmp;
            r1 =r1 + dr[i];
            c1 =c1 + dc[i];
        }
        if (tmp>0 && b.in_range(r1, c1) && b.at(r1, c1) ==player) {
            cnt += tmp;
        }
    }
    return cnt;
};

std::vector<std::pair<int,int>> legal_moves(const Board& b, int player) {
    std::vector<std::pair<int,int>> ret;
    for (int i=0;i<8;i++) {
        for (int j=0;j<8;j++) {
            if (flips_if_place(b, player, i, j) >0) {
                ret.push_back({i, j});
            }
        }
    }
    return ret;
};


void apply_move(Board& b, int player, int r, int c) {
    if (flips_if_place(b, player, r, c) ==0) {
        return;
    }
    b.at(r, c) =player;
    for (int i=0;i<8;i++) {
        int r1 =r + dr[i];
        int c1 =c + dc[i];
        int tmp =0;
        while (b.in_range(r1, c1) && b.at(r1, c1) == -player) {
            ++tmp;
            r1 =r1 + dr[i];
            c1 =c1 + dc[i];
        }
        if (tmp>0 && b.in_range(r1, c1) && b.at(r1, c1) ==player) {
            r1 = r +dr[i];
            c1 = c +dc[i];
            while (b.in_range(r1, c1) && b.at(r1, c1) == -player) {
                b.at(r1, c1) =player;
                r1 =r1 + dr[i];
                c1 =c1 + dc[i];
            }
        }
    }
};

bool has_any_move(const Board& b, int player) {
    for (int i =0;i<8;i++) {
        for (int j=0;j<8;j++) {
            if (flips_if_place(b, player, i, j) > 0) {
                return true;
            }
        }
    }
    return false;
};


bool is_game_over(const Board& b) {
    return !has_any_move(b, WHITE) && !has_any_move(b, BLACK);
};

int disc_diff(const Board& b) {
    int bc =0;
    int wc =0;
    for (int i=0;i<8;i++) {
        for (int j=0;j<8;j++) {
            if (b.at(i, j) ==WHITE) {
                wc ++;
            }
            else if (b.at(i, j) ==BLACK) {
                bc ++;
            }
        }
    }
    return bc -wc;
};// black_count - white_count

static constexpr int K = 5;

std::array<double, K> features(const Board& b, int player) {
    std::array<double, K> f ={0,0,0,0,0};
    int opp =-player;
    // f1:
    int me =0;
    int op =0;
    for (int i=0;i<8;i++) {
        for (int j=0;j<8;j++) {
            if (b.at(i, j) == player) {
                me ++;
            }
            else if (b.at(i, j) == opp) {
                op ++;
            }
        }
    }
    f[0] =(me -op) *1.0/64;
    
    int m_me = legal_moves(b, player).size();
    int m_op = legal_moves(b, opp).size();
    f[1] =(m_me -m_op) *1.0/(m_me + m_op +1);
    
    int four_corners[4][2] ={{0,0}, {0,7}, {7,0}, {7,7}};
    
    int cdiff_me =0;
    int cdiff_op =0;
    for (int i =0; i<4 ;i++) {
        if (b.at(four_corners[i][0], four_corners[i][1]) ==player) {
            cdiff_me++;
        }
        else if (b.at(four_corners[i][0], four_corners[i][1]) ==opp) {
            cdiff_op++;
        }
    }
    f[2] = cdiff_me - cdiff_op;
    
    int x_corners[4][2] ={{1,1}, {1,6}, {6,1}, {6,6}};
    int x_me =0;
    int x_op =0;
    for (int i=0;i<4;i++) {
        if (b.at(x_corners[i][0], x_corners[i][1]) ==player) {
            x_me++;
        }
        else if (b.at(x_corners[i][0], x_corners[i][1]) ==opp) {
            x_op++;
        }
    }
    f[3] = x_me - x_op;
    
    int edge_me =0;
    int edge_op =0;
    for (int i=0;i<8;i++) {
        if (b.at(0, i) == player) {
            edge_me ++;
        }
        else if (b.at(0, i) ==opp) {
            edge_op ++;
        }
        if (b.at(7, i) == player) {
            edge_me ++;
        }
        else if (b.at(7, i) ==opp) {
            edge_op ++;
        }
    }
    for (int i=1;i<7;i++) {
        if (b.at(i, 0) == player) {
            edge_me ++;
        }
        else if (b.at(i, 0) ==opp) {
            edge_op ++;
        }
        if (b.at(i, 7) == player) {
            edge_me ++;
        }
        else if (b.at(i, 7) ==opp) {
            edge_op ++;
        }
    }
    f[4] =(edge_me -edge_op)/28.0;

    return f;
};

double eval(const Board& b, int player, const std::array<double, K>& w) {
  auto f =features(b, player);
  double s =0;
  for (int i=0;i<K;i++) {
      s += w[i] * f[i];
  }
  return s;
};
const double INF =1e18;
double negamax(const Board& b, int player, int depth,
               double alpha, double beta,
               const std::array<double, K>& w) {
    if (is_game_over(b)) {
        int z =player == BLACK ? disc_diff(b) : -disc_diff(b);
        return (z>0  ? INF : (z<0 ? -INF : z)) + z;
    }
    if (depth ==0) {
        return eval(b, player, w);
    }              
    auto moves =legal_moves(b, player);
    if (moves.empty()) {
        return -negamax(b, -player, depth, -beta, -alpha, w);
    }
    double best =-INF;
    for (auto [r, c] : moves) {
        Board nb =b;
        apply_move(nb, player, r, c);
        double val = -negamax(nb, -player, depth-1, -beta, -alpha, w);
        best = std::max(best, val);
        alpha = std::max(alpha, val);
        if (alpha >= beta) {
            break;
        }
    }
    return best;
};

std::pair<int,int> best_move(const Board& b, int player, int depth,
                             const std::array<double, K>& w) {
    if (depth ==0) {
        depth =1;
    }
    auto moves =legal_moves(b, player);
    if (moves.empty()) {
        return {-1, -1};
    }
    double best =-INF;
    std::pair<int, int> ret ={-1, -1};
    for (auto [r,c] : moves) {
        Board nb = b; 
        apply_move(nb, player, r, c);
        double val = -negamax(nb, -player, depth-1, -INF, +INF, w);
        if (val  > best) {
            best =val;
            ret ={r, c};
        }
    }
    return ret;
};

struct GameResult {
    int final_disc_diff;   // black - white
    int winner;            // BLACK / WHITE / 0 (draw)
    int ply;               // 实际落子数（不含 pass）
    int passes;            // pass 次数
};

GameResult play_game(std::array<double,K> w_black,
                     std::array<double,K> w_white,
                     int depth_black,
                     int depth_white,
                     int max_plies = 200) {
    Board b;
    int player =BLACK;
    int passes =0;
    int ply =0;
    while (true) {
        if (is_game_over(b)) {
            break;
        }
        std::array<double,K> w_player;
        int depth_player;
        if (player ==BLACK) {
             w_player =w_black;
             depth_player =depth_black;
        }
        else {
            w_player = w_white;
            depth_player =depth_white;
        }
        auto mv = best_move(b, player, depth_player, w_player);
        if (mv == std::pair<int,int>{-1, -1}) {
            passes++;
            player =-player;
            continue;
        }
        apply_move(b, player, mv.first, mv.second);
        ply ++;
        if (ply > max_plies) {
            break;
        }
        player =-player;
    }
    GameResult gr;
    gr.final_disc_diff =disc_diff(b);
    gr.winner =gr.final_disc_diff > 0 ? BLACK :(gr.final_disc_diff <0 ? WHITE :0);
    gr.ply =ply;
    gr.passes =passes;
    return gr;
};

struct Step {
    Board b;               // 走子前局面（或走子后也行，但要一致）
    int player;            // 当前要走的一方
    std::pair<int,int> move; // (-1,-1) 表示 pass
};

struct GameTrace {
    std::vector<Step> steps;
    GameResult result;
};

GameTrace play_game_with_trace(std::array<double,K> w_black,
                               std::array<double,K> w_white,
                               int depth_black,
                               int depth_white,
                               int max_plies = 200) {
    Board b;
    int player =BLACK;
    int passes =0;
    int ply =0;
    GameTrace gt;
    while (true) {
        if (is_game_over(b)) {
            break;
        }
        std::array<double,K> w_player;
        int depth_player;
        if (player ==BLACK) {
             w_player =w_black;
             depth_player =depth_black;
        }
        else {
            w_player = w_white;
            depth_player =depth_white;
        }
        auto mv = best_move(b, player, depth_player, w_player);
        Step step;
        step.b =b;
        step.player =player;
        step.move =mv;
        gt.steps.push_back(step);
        
        if (mv == std::pair<int,int>{-1, -1}) {
            passes++;
            player =-player;
            continue;
        }
        apply_move(b, player, mv.first, mv.second);
        ply ++;
        if (ply > max_plies) {
            break;
        }
        player =-player;
    }
    GameResult gr;
    gr.final_disc_diff =disc_diff(b);
    gr.winner =gr.final_disc_diff > 0 ? BLACK :(gr.final_disc_diff <0 ? WHITE :0);
    gr.ply =ply;
    gr.passes =passes;
    gt.result =gr;
    return gt;                                   
};

std::pair<int,int> select_move_explore(const Board& b, int player,int depth, const std::array<double,K>& w, double epsilon) {
    auto moves = legal_moves(b, player);
    if (moves.empty()) {
        return {-1, -1};
    }
    static std::mt19937 rng((unsigned)std::time(nullptr));
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    if (uni01(rng) < epsilon) {
        std::uniform_int_distribution<int> pick(0, (int)moves.size() - 1);
        return moves[pick(rng)];
    }
    return best_move(b, player, depth, w);
};

struct Transition {
    std::array<double,K> f;   // features(s, player)
    std::array<double,K> f2;  // features(s', player')，若 terminal 则可全0
    double r;                 // reward：非终局0，终局±1/0
    bool terminal;
};

std::vector<Transition> make_transitions_from_trace(const GameTrace& gt, double gamma /*固定传1.0也行*/) {
    std::vector<Transition> transitions;
    for (auto step : gt.steps) {
        Transition tr;
        tr.f = features(step.b, step.player);
        Board b_next =step.b;
        if (step.move != std::pair<int, int>{-1, -1}) {
            apply_move(b_next, step.player, step.move.first, step.move.second);
        }
        int next_player = -step.player;
        if (!has_any_move(b_next, next_player) && !is_game_over(b_next)) {
          next_player = step.player;  // 对手无子可走，轮次不交换
        }

        bool terminal =is_game_over(b_next);
        tr.r =0;
        tr.f2 ={0 ,0 ,0, 0, 0};
        if (terminal) {
            int D = disc_diff(b_next);        // black - white
            int s = (D > 0) ? 1 : (D < 0 ? -1 : 0); // 黑胜=+1，白胜=-1，平=0
            tr.r =(step.player == BLACK) ? (double)s : (double)(-s);
        } else {
            tr.f2 =features(b_next, next_player);
        }
        transitions.push_back(tr);
    }
    return transitions;
};

void train_td_lambda(std::array<double,K>& w,
                     const std::vector<Transition>& traj,
                     double alpha,
                     double gamma,
                     double lambda) {
    std::array<double,K> e ={0 ,0, 0, 0, 0};                
    for (auto tran : traj) {
        double V =std::inner_product(tran.f.begin(), tran.f.end(), w.begin(), 0.0);
        double V2 =std::inner_product(tran.f2.begin(), tran.f2.end(), w.begin(), 0.0);
        double delta = tran.r + gamma * V2 - V;
        for (int i =0;i<K;i++) {
            e[i] =gamma * lambda * e[i] + tran.f[i];
        }
        for (int i =0;i<K;i++) {
            w[i] =w[i] + alpha * delta *e[i]; 
        }
    }
};

struct TrainConfig {
    int games;
    int depth;
    double alpha;
    double gamma;   // 1.0
    double lambda;  // 0.7~0.95
    double eps_start, eps_end;
};

void train_self_play(std::array<double,K>& w, const TrainConfig& cfg) {
    for (int i=0;i<cfg.games;i++) {
        double t = (cfg.games <= 1) ? 0.0 : (double)i / (cfg.games - 1);
        double epsilon = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * t;
        int player =BLACK;
        Board b;
        int max_plies =200;
        int passes =0;
        int ply =0;
        GameTrace gt;
        while (true) {
            if (is_game_over(b)) {
                break;
            }

            auto mv = select_move_explore(b, player, cfg.depth, w, epsilon);
            Step step;
            step.b =b;
            step.player =player;
            step.move =mv;
            gt.steps.push_back(step);
        
            if (mv == std::pair<int,int>{-1, -1}) {
                passes++;
                player =-player;
                continue;
            }
            apply_move(b, player, mv.first, mv.second);
            ply ++;
            if (ply > max_plies) {
                break;
            }
            player =-player;
        }
        GameResult gr;
        gr.final_disc_diff =disc_diff(b);
        gr.winner =gr.final_disc_diff > 0 ? BLACK :(gr.final_disc_diff <0 ? WHITE :0);
        gr.ply =ply;
        gr.passes =passes;
        gt.result =gr;
        auto traj = make_transitions_from_trace(gt, cfg.gamma);
        train_td_lambda(w, traj, cfg.alpha, cfg.gamma, cfg.lambda);
    }
};



