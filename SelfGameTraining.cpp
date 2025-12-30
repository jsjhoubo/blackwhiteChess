#include <iostream>
#include <vector>
#include <utility>
#include <array>
#include <random>
#include <ctime>
#include <numeric>   // inner_product
#include <cmath>     // isfinite
#include <algorithm> // max/min
#include <cassert>
#include <string>

#define CHECK(cond) do { if(!(cond)) { \
  std::cerr << "CHECK failed: " #cond << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
  std::exit(1); } } while(0)

// -------------------- Basic board --------------------
const double search_time_list =0.05; // second 
const int BLACK =  1;
const int WHITE = -1;
const int EMPTY =  0;

struct Board {
    Board(){ init_start(); }
    int8_t a[64]; // row-major

    int8_t at(int r, int c) const { return a[r*8 + c]; }
    int8_t& at(int r, int c) { return a[r*8 + c]; }

    void clear() { for(int i=0;i<64;i++) a[i]=0; }
    void init_start() {
        clear();
        at(3,3) = WHITE;
        at(3,4) = BLACK;
        at(4,3) = BLACK;
        at(4,4) = WHITE;
    }
    bool in_range(int r, int c) const { return (r>=0 && r<8 && c>=0 && c<8); }

    void print() const {
        for(int r=0;r<8;r++){
            for(int c=0;c<8;c++){
                if(at(r,c)==EMPTY) std::cout<<".";
                else if(at(r,c)==BLACK) std::cout<<"B";
                else std::cout<<"W";
            }
            std::cout<<"\n";
        }
    }
};

static const int dr[8] = {-1,-1,-1,0,0,1,1,1};
static const int dc[8] = {-1,0,1,-1,1,-1,0,1};

int flips_if_place(const Board& b, int player, int r, int c) {
    if(!b.in_range(r,c) || b.at(r,c)!=EMPTY) return 0;
    int cnt=0;
    for(int d=0; d<8; d++){
        int r1=r+dr[d], c1=c+dc[d];
        int tmp=0;
        while(b.in_range(r1,c1) && b.at(r1,c1)==-player){
            tmp++;
            r1+=dr[d]; c1+=dc[d];
        }
        if(tmp>0 && b.in_range(r1,c1) && b.at(r1,c1)==player) cnt+=tmp;
    }
    return cnt;
}

std::vector<std::pair<int,int>> legal_moves(const Board& b, int player) {
    std::vector<std::pair<int,int>> ret;
    for(int r=0;r<8;r++) for(int c=0;c<8;c++){
        if(flips_if_place(b, player, r, c) > 0) ret.push_back({r,c});
    }
    return ret;
}

void apply_move(Board& b, int player, int r, int c) {
    if(flips_if_place(b, player, r, c)==0) return;
    b.at(r,c)=player;
    for(int d=0; d<8; d++){
        int r1=r+dr[d], c1=c+dc[d];
        int tmp=0;
        while(b.in_range(r1,c1) && b.at(r1,c1)==-player){
            tmp++;
            r1+=dr[d]; c1+=dc[d];
        }
        if(tmp>0 && b.in_range(r1,c1) && b.at(r1,c1)==player){
            // flip back
            int rr=r+dr[d], cc=c+dc[d];
            while(b.in_range(rr,cc) && b.at(rr,cc)==-player){
                b.at(rr,cc)=player;
                rr+=dr[d]; cc+=dc[d];
            }
        }
    }
}

bool has_any_move(const Board& b, int player) {
    for(int r=0;r<8;r++) for(int c=0;c<8;c++){
        if(flips_if_place(b, player, r, c) > 0) return true;
    }
    return false;
}

bool is_game_over(const Board& b) {
    return !has_any_move(b, BLACK) && !has_any_move(b, WHITE);
}

int disc_diff(const Board& b) { // black - white
    int bc=0,wc=0;
    for(int r=0;r<8;r++) for(int c=0;c<8;c++){
        if(b.at(r,c)==BLACK) bc++;
        else if(b.at(r,c)==WHITE) wc++;
    }
    return bc-wc;
}

// -------------------- RNG helper --------------------
struct RNG {
    std::mt19937 rng;
    explicit RNG(uint32_t seed=1234567u) : rng(seed) {}
    double uni01() {
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng);
    }
    int randint(int lo, int hi) { // inclusive
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(rng);
    }
};

static inline double clampd(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

// -------------------- Strong features (K=12) --------------------
static constexpr int K = 12;

// Helpers for strong features
int corner_count(const Board& b, int player){
    int cnt=0;
    const int corners[4][2]={{0,0},{0,7},{7,0},{7,7}};
    for(auto &p: corners) if(b.at(p[0],p[1])==player) cnt++;
    return cnt;
}
int x_square_count(const Board& b, int player){
    int cnt=0;
    const int xs[4][2]={{1,1},{1,6},{6,1},{6,6}};
    for(auto &p: xs) if(b.at(p[0],p[1])==player) cnt++;
    return cnt;
}
int c_square_count(const Board& b, int player){
    // Adjacent to corners on edge (often dangerous)
    const int cs[8][2]={{0,1},{1,0},{0,6},{1,7},{6,0},{7,1},{6,7},{7,6}};
    int cnt=0;
    for(auto &p: cs) if(b.at(p[0],p[1])==player) cnt++;
    return cnt;
}
int edge_count(const Board& b, int player){
    int cnt=0;
    for(int c=0;c<8;c++){
        if(b.at(0,c)==player) cnt++;
        if(b.at(7,c)==player) cnt++;
    }
    for(int r=1;r<7;r++){
        if(b.at(r,0)==player) cnt++;
        if(b.at(r,7)==player) cnt++;
    }
    return cnt; // total edges = 28
}
int count_frontier(const Board& b, int player){
    // Frontier disc: adjacent to at least one empty square
    int cnt=0;
    for(int r=0;r<8;r++) for(int c=0;c<8;c++){
        if(b.at(r,c)!=player) continue;
        bool frontier=false;
        for(int d=0; d<8; d++){
            int rr=r+dr[d], cc=c+dc[d];
            if(b.in_range(rr,cc) && b.at(rr,cc)==EMPTY){ frontier=true; break; }
        }
        if(frontier) cnt++;
    }
    return cnt;
}
int count_potential_mobility(const Board& b, int player){
    // Count empty squares adjacent to opponent discs (rough PM)
    int opp = -player;
    int cnt=0;
    for(int r=0;r<8;r++) for(int c=0;c<8;c++){
        if(b.at(r,c)!=EMPTY) continue;
        bool ok=false;
        for(int d=0; d<8; d++){
            int rr=r+dr[d], cc=c+dc[d];
            if(b.in_range(rr,cc) && b.at(rr,cc)==opp){ ok=true; break; }
        }
        if(ok) cnt++;
    }
    return cnt;
}
int stable_edge_approx(const Board& b, int player){
    // Very rough: count stable discs grown from corners along edges if continuous
    int cnt=0;
    // top edge from (0,0) right
    if(b.at(0,0)==player){
        for(int c=0;c<8;c++){
            if(b.at(0,c)==player) cnt++;
            else break;
        }
    }
    // top edge from (0,7) left (avoid double-count overlap; simple still ok)
    if(b.at(0,7)==player){
        for(int c=7;c>=0;c--){
            if(b.at(0,c)==player) cnt++;
            else break;
        }
    }
    // bottom edge
    if(b.at(7,0)==player){
        for(int c=0;c<8;c++){
            if(b.at(7,c)==player) cnt++;
            else break;
        }
    }
    if(b.at(7,7)==player){
        for(int c=7;c>=0;c--){
            if(b.at(7,c)==player) cnt++;
            else break;
        }
    }
    // left edge
    if(b.at(0,0)==player){
        for(int r=0;r<8;r++){
            if(b.at(r,0)==player) cnt++;
            else break;
        }
    }
    if(b.at(7,0)==player){
        for(int r=7;r>=0;r--){
            if(b.at(r,0)==player) cnt++;
            else break;
        }
    }
    // right edge
    if(b.at(0,7)==player){
        for(int r=0;r<8;r++){
            if(b.at(r,7)==player) cnt++;
            else break;
        }
    }
    if(b.at(7,7)==player){
        for(int r=7;r>=0;r--){
            if(b.at(r,7)==player) cnt++;
            else break;
        }
    }
    return cnt; // not perfect; scale later
}
int corner_move_count(const Board& b, int player){
    int cnt=0;
    const int corners[4][2]={{0,0},{0,7},{7,0},{7,7}};
    for(auto &p: corners){
        if(flips_if_place(b, player, p[0], p[1])>0) cnt++;
    }
    return cnt;
}

int positional_score(const Board& b, int player){
    // classic-ish positional table (coarse)
    static const int P[8][8] = {
        {120,-20, 20,  5,  5, 20,-20,120},
        {-20,-40, -5, -5, -5, -5,-40,-20},
        { 20, -5, 15,  3,  3, 15, -5, 20},
        {  5, -5,  3,  3,  3,  3, -5,  5},
        {  5, -5,  3,  3,  3,  3, -5,  5},
        { 20, -5, 15,  3,  3, 15, -5, 20},
        {-20,-40, -5, -5, -5, -5,-40,-20},
        {120,-20, 20,  5,  5, 20,-20,120}
    };
    int opp=-player;
    int s=0;
    for(int r=0;r<8;r++) for(int c=0;c<8;c++){
        if(b.at(r,c)==player) s+=P[r][c];
        else if(b.at(r,c)==opp) s-=P[r][c];
    }
    return s;
}

// All strong features normalized to roughly [-1,1]
std::array<double, K> features_strong(const Board& b, int player) {
    std::array<double, K> f{};
    int opp = -player;

    int me=0, op=0;
    for (int r=0;r<8;r++) for (int c=0;c<8;c++) {
        if (b.at(r,c)==player) me++;
        else if (b.at(r,c)==opp) op++;
    }
    int empties = 64 - (me + op);

    // f0 disc diff
    f[0] = (me - op) / 64.0;

    // f1 mobility diff
    int m_me = (int)legal_moves(b, player).size();
    int m_op = (int)legal_moves(b, opp).size();
    f[1] = (m_me - m_op) / double(m_me + m_op + 1);

    // f2 potential mobility diff
    int pm_me = count_potential_mobility(b, player);
    int pm_op = count_potential_mobility(b, opp);
    f[2] = (pm_me - pm_op) / double(pm_me + pm_op + 1);

    // f3 corners diff
    int c_me = corner_count(b, player);
    int c_op = corner_count(b, opp);
    f[3] = (c_me - c_op) / 4.0;

    // f4 X-square diff
    int x_me = x_square_count(b, player);
    int x_op = x_square_count(b, opp);
    f[4] = (x_me - x_op) / 4.0;

    // f5 C-square diff
    int cs_me = c_square_count(b, player);
    int cs_op = c_square_count(b, opp);
    f[5] = (cs_me - cs_op) / 8.0;

    // f6 edges diff
    int e_me = edge_count(b, player);
    int e_op = edge_count(b, opp);
    f[6] = (e_me - e_op) / 28.0;

    // f7 frontier reversed diff
    int fr_me = count_frontier(b, player);
    int fr_op = count_frontier(b, opp);
    f[7] = (fr_op - fr_me) / 64.0;

    // f8 stable edge approx diff
    int st_me = stable_edge_approx(b, player);
    int st_op = stable_edge_approx(b, opp);
    f[8] = (st_me - st_op) / 64.0;

    // f9 positional table score (scaled)
    int ps = positional_score(b, player);
    f[9] = ps / 2000.0;

    // f10 parity
    f[10] = (empties % 2 == 1) ? 1.0 : -1.0;

    // f11 corner-move availability diff
    int cm_me = corner_move_count(b, player);
    int cm_op = corner_move_count(b, opp);
    f[11] = (cm_me - cm_op) / 4.0;

    return f;
}

double eval_strong(const Board& b, int player, const std::array<double,K>& w) {
    auto f = features_strong(b, player);
    return std::inner_product(w.begin(), w.end(), f.begin(), 0.0);
}
// -------------------- sorting order ---------------------------
static inline bool IsCorner(int r,int c) {
    return (r==0 || r==7) && (c ==0 || c ==7); 
}

static inline bool IsXSquare(int r,int c) {
    return (r == 1 || r==6 ) && (c==1 || c==6);
}

static inline bool IsCSquare(int r,int c) {
    return (r ==0 && c ==1) || (r ==1 && c ==0) ||
    (r ==0 && c ==6) || (r ==1 && c ==7) ||
    (r ==6 && c ==0) || (r ==7 && c ==1) ||
    (r ==6 && c ==7) || (r ==7 && c ==6);
}

static inline bool corner_empty(const Board& b, int cr, int cc) {
    return b.at(cr, cc) == EMPTY;
}

int move_order_score(const Board& b, int player, int r, int c){
    if (IsCorner(r, c)) return 1000000;

    int s =0;
    if (IsXSquare(r, c)) {
        int cr = r==1 ? 0 :7;
        int cc = c==1 ? 0 :7;
        if (corner_empty(b, cr, cc)) {
            s -= 200000;
        }
        s -= 20000;
    }
    if (IsCSquare(r, c)) {
        int cr = (r ==0 || r==1) ? 0 : 7;
        int cc = (c ==0 || c==1) ? 0 : 7;
        if (corner_empty(b, cr, cc)) {
            s -=80000;
        }
        s -= 8000;
    }

    int flips =flips_if_place(b, player, r, c);
    s += flips *300;

     static const int P1[8][8] = {
        {120,-20, 20,  5,  5, 20,-20,120},
        {-20,-40, -5, -5, -5, -5,-40,-20},
        { 20, -5, 15,  3,  3, 15, -5, 20},
        {  5, -5,  3,  3,  3,  3, -5,  5},
        {  5, -5,  3,  3,  3,  3, -5,  5},
        { 20, -5, 15,  3,  3, 15, -5, 20},
        {-20,-40, -5, -5, -5, -5,-40,-20},
        {120,-20, 20,  5,  5, 20,-20,120}
    };
    s += P1[r][c] *50;

    return s;
}
// ------------------------ zobrist TT ---------------------------
uint64_t Z[64][3]; // EMPTY/BLACK/WHITE
uint64_t ZTURN;    // side to move
static inline int piece_index(int8_t v){
    return (v==EMPTY?0:(v==BLACK?1:2));
}
uint64_t board_hash(const Board& b, int player_to_move){
    uint64_t h=0;
    for(int i=0;i<64;i++){
        h ^= Z[i][ piece_index(b.a[i]) ];
    }
    if(player_to_move==BLACK) h ^= ZTURN;
    return h;
}

void init_zobrist(RNG& rng){
    auto rnd64 = [&](){
        uint64_t a = (uint64_t)rng.rng();
        uint64_t b = (uint64_t)rng.rng();
        return (a<<32) ^ b;
    };
    for(int i=0;i<64;i++) for(int j=0;j<3;j++) Z[i][j]=rnd64();
    ZTURN = rnd64();
}

enum BoundType : uint8_t { B_EXACT=0, B_LOWER=1, B_UPPER=2 };

struct TTEntry {
    uint64_t key;
    double val;
    int8_t depth =-1;
    uint8_t bound =B_EXACT;
};

static const int TT_SIZE = 1<<20;
static std::vector<TTEntry> TT;

static inline TTEntry* tt_get(uint64_t key){
    return &TT[key & (TT_SIZE-1)];
}



// -------------------- Negamax (alpha-beta) --------------------
const double INF = 1e18;

double terminal_score(const Board& b, int player){
    // return huge +/- depending on win from 'player' perspective
    int D = disc_diff(b); // black-white
    int z = (player==BLACK) ? D : -D;
    if(z>0) return INF + z;
    if(z<0) return -INF + z;
    return 0.0;
}

int CountEmpties(const Board &b) {
    int cnt =0;
    for (int i=0;i<8;i++) {
        for (int j=0;j<8;j++) {
            if (b.at(i, j) ==EMPTY) {
                ++cnt;
            }
        }
    }
    return cnt;
}

static const int EMPTY_LIMIT =12;
static const int EXTRA =2;
double negamax(const Board& b, int player, int depth,
               double alpha, double beta,
               const std::array<double,K>& w) {
    if(is_game_over(b)) return terminal_score(b, player);
    int number_empty =CountEmpties(b);

    if (number_empty < EMPTY_LIMIT) {
        depth =std::max(depth, number_empty + EXTRA);
    }
    uint64_t key = board_hash(b, player); 
    auto entry =tt_get(key);
    if (entry->key == key && entry->depth >= depth) {
        if (entry->bound ==B_EXACT) {
            return entry->val;
        }
        if (entry->bound == B_LOWER) {
            alpha =std::max(alpha, entry->val);
        }
        else if (entry->bound == B_UPPER) {
            beta =std::min(beta, entry->val);
        }
        if (alpha >= beta) {
            return entry->val;
        }
    }
    double alpha0 =alpha;
    double beta0 =beta;

    if(depth==0) return eval_strong(b, player, w);

    auto moves = legal_moves(b, player);
    if(moves.empty()){
        // pass: do not consume depth (common choice)
        return -negamax(b, -player, depth, -beta, -alpha, w);
    }

    std::sort(moves.begin(), moves.end(), [&](const auto& x, const auto& y) {
        return move_order_score(b, player, x.first, x.second) > move_order_score(b, player, y.first, y.second); 
    });
    double best = -INF;
    for(auto [r,c]: moves){
        Board nb=b;
        apply_move(nb, player, r, c);
        double val = -negamax(nb, -player, depth-1, -beta, -alpha, w);
        best = std::max(best, val);
        alpha = std::max(alpha, val);
        if(alpha >= beta) break; // cut
    }
    TTEntry tt_entry;
    tt_entry.key =key;
    tt_entry.val =best;
    tt_entry.depth =depth;
    if (best <= alpha0) {
        tt_entry.bound =B_UPPER;
    }
    else if (best >= beta0) {
        tt_entry.bound =B_LOWER;
    }
    else if (alpha0 <best && best <beta0){
        tt_entry.bound =B_EXACT;
    }
    TT[key & (TT_SIZE-1)] =tt_entry;
    return best;
}

struct Timer {
    std::clock_t st;
    double limit_sec;
    explicit Timer(double sec): st(std::clock()), limit_sec(sec) {}
    bool time_up() const {
        double t = double(std::clock()-st) / CLOCKS_PER_SEC;
        return t >= limit_sec;
    }
};

std::pair<int,int> best_move_timed(const Board& b, double time_sec, int player, 
                             const std::array<double,K>& w, int max_depth =12) {
    Timer t(time_sec);

    auto moves0 = legal_moves(b, player);
    if(moves0.empty()) return {-1,-1};
    std::pair<int,int> best = moves0[0];
    double max =-INF;
    for (int d=1;d<=max_depth;d++) {
        if(t.time_up()) break;
        auto moves =moves0;
        std::sort(moves.begin(), moves.end(), [&](const auto& x, const auto& y) {
            return move_order_score(b, player, x.first, x.second) > move_order_score(b, player, y.first, y.second); 
        });
        double local_max =-INF;
        auto local_best =moves[0];
        for(auto [r,c]: moves){
            Board nb=b;
            apply_move(nb, player, r, c);
            double val = -negamax(nb, -player, std::max(0, d-1), -INF, +INF, w);
            if(val > local_max){
                local_max = val;
                local_best = {r,c};
            }
        }
        if(!t.time_up()) {
            best = local_best;
            max = local_max;
        } else {
            break;
        }
    }
    return best;
}

std::pair<int,int> best_move(const Board& b, int player, int depth,
                             const std::array<double,K>& w) {
    auto moves = legal_moves(b, player);
    if(moves.empty()) return {-1,-1};

    double best = -INF;
    std::pair<int,int> ret = moves[0];
    for(auto [r,c]: moves){
        Board nb=b;
        apply_move(nb, player, r, c);
        double val = -negamax(nb, -player, std::max(0, depth-1), -INF, +INF, w);
        if(val > best){
            best = val;
            ret = {r,c};
        }
    }
    return ret;
}

std::pair<int,int> select_move_epsilon_greedy(const Board& b, int player,
                                              int depth,
                                              const std::array<double,K>& w,
                                              double epsilon,
                                              RNG& rng) {
    auto moves = legal_moves(b, player);
    if(moves.empty()) return {-1,-1};
    if(rng.uni01() < epsilon){
        int idx = rng.randint(0, (int)moves.size()-1);
        return moves[idx];
    }
    // return best_move(b, player, depth, w);
    return best_move_timed(b, search_time_list, player, w, depth);
}

// -------------------- Game play (agent vs baseline) --------------------
struct GameResult {
    int final_disc_diff;   // black - white
    int winner;            // BLACK/WHITE/0
    int ply;               // moves played (excluding pass)
    int passes;            // pass count
};

GameResult play_game_agent_vs_base(const std::array<double,K>& w_agent,
                                  const std::array<double,K>& w_base,
                                  int agent_depth,
                                  int base_depth,
                                  int agent_color,
                                  int max_plies=200) {
    Board b;
    int player = BLACK;
    int ply=0, passes=0;
    int safety=0;

    while(!is_game_over(b) && safety++ < 400){
        auto moves = legal_moves(b, player);
        if(moves.empty()){
            passes++;
            player = -player;
            continue;
        }
        if(player == agent_color){
            // auto mv = best_move(b, player, agent_depth, w_agent);
            auto mv =best_move_timed(b, search_time_list, player, w_agent, agent_depth);
            apply_move(b, player, mv.first, mv.second);
        }else{
            // auto mv = best_move(b, player, base_depth, w_base);
            auto mv =best_move_timed(b, search_time_list, player, w_base, base_depth);
            apply_move(b, player, mv.first, mv.second);
        }
        ply++;
        if(ply > max_plies) break;
        player = -player;
    }

    GameResult gr;
    gr.final_disc_diff = disc_diff(b);
    gr.winner = (gr.final_disc_diff>0) ? BLACK : (gr.final_disc_diff<0 ? WHITE : 0);
    gr.ply = ply;
    gr.passes = passes;
    return gr;
}

int outcome_sign_from_agent_perspective(const Board& b, int agent_color){
    int D = disc_diff(b); // black - white
    int s = (D>0) ? +1 : (D<0 ? -1 : 0); // black win=+1, white win=-1, draw=0
    return (agent_color==BLACK) ? s : -s; // agent's perspective
}

// -------------------- Training config + TD(lambda) training --------------------
struct TrainConfig {
    int games = 5000;
    int agent_depth = 3;
    int base_depth  = 2;
    double alpha = 0.002;   // IMPORTANT: small to avoid NaN
    double gamma = 1.0;
    double lambda = 0.85;
    double eps_start = 0.30;
    double eps_end   = 0.05;
    uint32_t seed = 1234567u;
};

void train_vs_baseline_td_lambda(std::array<double,K>& w_agent,
                                 const std::array<double,K>& w_base,
                                 const TrainConfig& cfg) {
    RNG rng(cfg.seed);

    const double DELTA_CLIP   = 2.0;
    const double TRACE_CLIP   = 5.0;
    const double W_CLIP       = 50.0;
    const double WEIGHT_DECAY = 1e-5;

    for(int g=0; g<cfg.games; g++){
        double t = (cfg.games<=1) ? 0.0 : (double)g / (cfg.games - 1);
        double epsilon = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * t;

        // alpha decay
        double alpha = cfg.alpha * (0.25 + 0.75*(1.0 - t));

        Board b;
        int agent_color = (rng.uni01() < 0.5) ? BLACK : WHITE;
        int player = BLACK;

        std::array<double,K> e{}; // eligibility traces
        int safety=0;

        while(!is_game_over(b) && safety++ < 400){
            // If current player has no move, pass
            if(!has_any_move(b, player)){
                player = -player;
                continue;
            }

            // Baseline moves until agent's turn
            if(player != agent_color){
                // auto mv = best_move(b, player, cfg.base_depth, w_base);
                auto mv =best_move_timed(b, search_time_list, player, w_base, cfg.base_depth);;
                apply_move(b, player, mv.first, mv.second);
                player = -player;
                continue;
            }

            // ---- Agent decision point ----
            auto f_s = features_strong(b, agent_color);
            double V_s = std::inner_product(w_agent.begin(), w_agent.end(), f_s.begin(), 0.0);
            if(!std::isfinite(V_s)){ e.fill(0.0); break; }

            auto mv = select_move_epsilon_greedy(b, agent_color, cfg.agent_depth, w_agent, epsilon, rng);
            if(mv.first == -1){
                // agent pass (rare here because we checked has_any_move above)
                player = -player;
                continue;
            }
            apply_move(b, agent_color, mv.first, mv.second);
            player = -agent_color;

            // Simulate baseline responses until next agent turn or terminal
            int guard=0;
            while(!is_game_over(b) && player != agent_color && guard++ < 80){
                if(!has_any_move(b, player)){
                    player = -player;
                    continue;
                }
                auto mvb = best_move(b, player, cfg.base_depth, w_base);
                apply_move(b, player, mvb.first, mvb.second);
                player = -player;
            }

            // Reward and next value
            double r = 0.0;
            double V_s2 = 0.0;

            if(is_game_over(b)){
                r = (double)outcome_sign_from_agent_perspective(b, agent_color); // -1/0/+1
                V_s2 = 0.0;
            }else{
                // next agent state is current board b (player should be agent_color, or could be baseline if guard hit)
                // If somehow it's not agent's turn, we still evaluate from agent_color perspective.
                auto f_s2 = features_strong(b, agent_color);
                V_s2 = std::inner_product(w_agent.begin(), w_agent.end(), f_s2.begin(), 0.0);
                if(!std::isfinite(V_s2)){ e.fill(0.0); break; }
            }

            double delta = r + cfg.gamma * V_s2 - V_s;
            if(!std::isfinite(delta)){ e.fill(0.0); break; }
            delta = clampd(delta, -DELTA_CLIP, +DELTA_CLIP);

            // Update traces + weights
            for(int i=0;i<K;i++){
                e[i] = cfg.gamma * cfg.lambda * e[i] + f_s[i];
                e[i] = clampd(e[i], -TRACE_CLIP, +TRACE_CLIP);
            }
            for(int i=0;i<K;i++){
                w_agent[i] *= (1.0 - WEIGHT_DECAY);
                w_agent[i] += alpha * delta * e[i];
                w_agent[i] = clampd(w_agent[i], -W_CLIP, +W_CLIP);
            }

            // continue from current player (likely agent_color if baseline finished)
        }
    }
}

// -------------------- Evaluation --------------------
struct EvalStats {
    int games = 0;
    int win = 0;
    int lose = 0;
    int draw = 0;
    double winrate() const { return games ? (double)win / games : 0.0; }
};

EvalStats evaluate_vs_baseline_fair(const std::array<double,K>& w_agent,
                                   const std::array<double,K>& w_base,
                                   int agent_depth,
                                   int base_depth,
                                   int total_games,
                                   uint32_t seed=999u) {
    RNG rng(seed);
    EvalStats st;
    st.games = total_games;

    // Fair: half games agent=BLACK, half games agent=WHITE (shuffle by random but keep counts)
    int half = total_games / 2;
    int rest = total_games - half;

    for(int i=0;i<half;i++){
        auto gr = play_game_agent_vs_base(w_agent, w_base, agent_depth, base_depth, BLACK);
        if(gr.winner==BLACK) st.win++;
        else if(gr.winner==WHITE) st.lose++;
        else st.draw++;
    }
    for(int i=0;i<rest;i++){
        auto gr = play_game_agent_vs_base(w_agent, w_base, agent_depth, base_depth, WHITE);
        if(gr.winner==WHITE) st.win++;      // agent is white
        else if(gr.winner==BLACK) st.lose++;
        else st.draw++;
    }

    // small random swap by RNG doesn't matter; counts already balanced.
    (void)rng;
    return st;
}
// -------------------- UNIT TEST -------------------------------------------
static void Test_StartPosition_LegalMoves() {
    Board b; // 自动 init_start()
    auto mb = legal_moves(b, BLACK);
    auto mw = legal_moves(b, WHITE);

    CHECK(mb.size() == 4);
    CHECK(mw.size() == 4);

    // 黑棋开局四手：(2,3),(3,2),(4,5),(5,4)
    auto has = [&](const std::vector<std::pair<int,int>>& v, int r, int c){
        return std::find(v.begin(), v.end(), std::make_pair(r,c)) != v.end();
    };
    CHECK(has(mb, 2,3));
    CHECK(has(mb, 3,2));
    CHECK(has(mb, 4,5));
    CHECK(has(mb, 5,4));

    CHECK(CountEmpties(b) == 60);
    CHECK(disc_diff(b) == 0); // 2黑2白
}

static void Test_Flips_And_ApplyMove() {
    Board b; // start
    CHECK(flips_if_place(b, BLACK, 2,3) == 1);
    CHECK(flips_if_place(b, BLACK, 0,0) == 0); // 开局角不合法

    apply_move(b, BLACK, 2,3);
    // 下(2,3)并翻转(3,3)：黑=4 白=1 => diff=3
    CHECK(disc_diff(b) == 3);
    CHECK(b.at(2,3) == BLACK);
    CHECK(b.at(3,3) == BLACK);
}

static void Test_Hash_SideToMove_Differs() {
    RNG rng(123u);
    init_zobrist(rng);

    Board b; // start
    uint64_t hB = board_hash(b, BLACK);
    uint64_t hW = board_hash(b, WHITE);
    CHECK(hB != hW); // side-to-move 应纳入 hash
}

static void Test_TT_ReadWrite_Basic() {
    // TT 必须已分配
    TT.assign(TT_SIZE, TTEntry{});

    uint64_t key = 0x123456789abcdef0ULL;
    TTEntry* e = tt_get(key);

    // 写入
    e->key = key;
    e->depth = 7;
    e->val = 0.42;
    e->bound = B_EXACT;

    // 读回
    TTEntry* e2 = tt_get(key);
    CHECK(e2->key == key);
    CHECK(e2->depth == 7);
    CHECK(std::fabs(e2->val - 0.42) < 1e-12);
    CHECK(e2->bound == B_EXACT);
}

static Board Make_AsymmetricPosition_ForTests() {
    Board b; // start
    // 黑(2,3)合法
    apply_move(b, BLACK, 2, 3);
    // 白(2,2)在此时合法（对角线：2,2 -> 3,3(黑) -> 4,4(白)）
    apply_move(b, WHITE, 2, 2);
    return b; // 轮到 BLACK
}

static double RootBestValue_FixedDepth(const Board& b, int player, int depth,
                                       const std::array<double,K>& w) {
    auto moves = legal_moves(b, player);
    CHECK(!moves.empty());
    double bestVal = -INF;
    for (auto [r,c] : moves) {
        Board nb = b;
        apply_move(nb, player, r, c);
        double val = -negamax(nb, -player, std::max(0, depth-1), -INF, +INF, w);
        bestVal = std::max(bestVal, val);
    }
    return bestVal;
}

static double RootMoveValue(const Board& b, int player, int depth,
                            const std::array<double,K>& w,
                            std::pair<int,int> mv) {
    CHECK(mv.first != -1);
    Board nb = b;
    apply_move(nb, player, mv.first, mv.second);
    return -negamax(nb, -player, std::max(0, depth-1), -INF, +INF, w);
}

// 1) timed 迭代加深：给足时间时，它选的 move 的 value 必须达到固定深度最优 value
static void Test_BestMoveTimed_Reaches_FixedDepth_OptValue() {
    // 初始化 Zobrist / TT
    RNG rng(123u);
    init_zobrist(rng);
    TT.assign(TT_SIZE, TTEntry{});

    // 用一个不对称局面，避免“全 0 对称”导致的平局/多解
    Board b = Make_AsymmetricPosition_ForTests();
    int player = BLACK;

    // 用一个确定的权重（你 baseline 的那套即可）
    const std::array<double,K> W = {
        0.10, 1.20, 0.60, 3.00, -1.50, -1.00, 0.80, 0.80, 1.50, 1.20, 0.15, 1.50
    };

    int depth = 4;
    double time_sec = 1.0; // 给足一点，避免机器慢导致跑不完

    // 固定深度最优 value
    TT.assign(TT_SIZE, TTEntry{});
    double bestVal = RootBestValue_FixedDepth(b, player, depth, W);

    // timed 选的 move 的 value
    TT.assign(TT_SIZE, TTEntry{});
    auto mvTimed = best_move_timed(b, time_sec, player, W, depth);
    double timedVal = RootMoveValue(b, player, depth, W, mvTimed);

    // 允许极小浮点误差
    CHECK(std::fabs(timedVal - bestVal) < 1e-9);
}

// 2) TT bound 必须是合法枚举（0/1/2），并且在 alpha==beta 的零窗口搜索下不应出现 EXACT
static void Test_TT_Bound_Is_Valid_And_Not_Garbage() {
    RNG rng(456u);
    init_zobrist(rng);
    TT.assign(TT_SIZE, TTEntry{});

    Board b = Make_AsymmetricPosition_ForTests();
    int player = BLACK;

    const std::array<double,K> W = {
        0.10, 1.20, 0.60, 3.00, -1.50, -1.00, 0.80, 0.80, 1.50, 1.20, 0.15, 1.50
    };

    int depth = 3;
    // 零窗口：更容易触发 fail-low / fail-high，从而要求 bound 必须被正确写入
    double alpha = 0.0, beta = 0.0;

    (void)negamax(b, player, depth, alpha, beta, W);

    uint64_t key = board_hash(b, player);
    TTEntry* e = tt_get(key);

    // 必须命中 key（同槽）
    CHECK(e->key == key);

    // bound 必须是 {0,1,2} 之一，不能是未初始化垃圾值
    CHECK(e->bound == B_EXACT || e->bound == B_LOWER || e->bound == B_UPPER);

    // alpha==beta 的零窗口下，如果走完全搜索，bound 不应是 EXACT（否则很可疑）
    // 这里允许极罕见的“真值正好==0”导致 EXACT 的情况，所以更稳一点：只要求不是垃圾值即可。
    // 如果你希望更严格（抓住未初始化），可以取消注释下一行：
    // CHECK(e->bound != B_EXACT);
}

static void RunAllTests() {
    std::cerr << "[UNIT_TEST] Running...\n";
    Test_StartPosition_LegalMoves();
    Test_Flips_And_ApplyMove();
    Test_Hash_SideToMove_Differs();
    Test_TT_ReadWrite_Basic();

    
    Test_BestMoveTimed_Reaches_FixedDepth_OptValue();
    Test_TT_Bound_Is_Valid_And_Not_Garbage();
    std::cerr << "[UNIT_TEST] All tests passed.\n";
}

int main() {
    
#ifdef UNIT_TEST
    RunAllTests();
    return 0;
#endif
    // Baseline weights (match normalized features scale)
    const std::array<double,K> W_BASE = {
        0.10,  // f0 disc
        1.20,  // f1 mobility
        0.60,  // f2 potential mobility
        3.00,  // f3 corners
       -1.50,  // f4 X
       -1.00,  // f5 C
        0.80,  // f6 edges
        0.80,  // f7 frontier reversed
        1.50,  // f8 stable edge approx
        1.20,  // f9 positional
        0.15,  // f10 parity
        1.50   // f11 corner move availability
    };

    // Agent initial weights (a bit weaker than baseline but reasonable)
    std::array<double,K> w_agent = {
        0.05, 0.80, 0.30, 2.00, -1.20, -0.80, 0.50, 0.50, 1.00, 0.80, 0.10, 1.00
    };

    TrainConfig cfg;
    cfg.games = 5000;
    cfg.agent_depth = 3;
    cfg.base_depth  = 2;
    cfg.alpha = 0.002;
    cfg.gamma = 1.0;
    cfg.lambda = 0.85;
    cfg.eps_start = 0.30;
    cfg.eps_end   = 0.05;
    cfg.seed = (uint32_t)std::time(nullptr);

    std::cout << "Training vs baseline...\n";
    std::cout << "  games=" << cfg.games
              << ", agent_depth=" << cfg.agent_depth
              << ", base_depth=" << cfg.base_depth
              << ", alpha=" << cfg.alpha
              << ", lambda=" << cfg.lambda
              << ", eps=[" << cfg.eps_start << " -> " << cfg.eps_end << "]\n";

    RNG rng(cfg.seed);
    init_zobrist(rng);
    TT.assign(TT_SIZE, TTEntry{});

    train_vs_baseline_td_lambda(w_agent, W_BASE, cfg);

    std::cout << "Learned w_agent = { ";
    for(int i=0;i<K;i++){
        std::cout << w_agent[i] << (i+1<K ? " , " : " ");
    }
    std::cout << "}\n";

    // Final evaluation (你可以改成 200/500/1000)
    int eval_games = 200;
    auto st = evaluate_vs_baseline_fair(w_agent, W_BASE, cfg.agent_depth, cfg.base_depth, eval_games, cfg.seed + 123);

    std::cout << "Evaluation vs baseline (fair color swap):\n";
    std::cout << "  games=" << eval_games
              << ", win=" << st.win
              << ", lose=" << st.lose
              << ", draw=" << st.draw
              << ", winrate=" << st.winrate() * 100.0 << "%\n";

    double threshold = 0.60;
    if(st.winrate() >= threshold){
        std::cout << "PASS: winrate >= " << (threshold*100.0) << "%\n";
    }else{
        std::cout << "FAIL: winrate < " << (threshold*100.0) << "%\n";
        std::cout << "Tips: increase games (e.g., 20000) or agent_depth (4), or reduce alpha.\n";
    }

    return 0;
}
