// Level 8 FINAL: Othello/Reversi engine + Negamax baseline + TD(λ) training vs baseline
// What this file does (in main):
// 1) Runs deterministic unit tests (rules + pass behavior).
// 2) Trains 5000 games with depth=3 (agent) vs baseline (depth=2).
// 3) Evaluates win rate vs baseline with color swap. Pass condition: >=60% => "LEVEL 8 PASSED".
//
// Compile: g++ -O2 -std=c++17 main.cpp && ./a.out
// (Online compilers: just paste and Run)

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

static constexpr int BLACK = 1;
static constexpr int WHITE = -1;
static constexpr int EMPTY = 0;

struct Board {
    int8_t a[64]; // row-major

    Board() { init_start(); }

    int8_t at(int r, int c) const { return a[r * 8 + c]; }
    int8_t& at(int r, int c) { return a[r * 8 + c]; }

    bool in_range(int r, int c) const { return (r >= 0 && r < 8 && c >= 0 && c < 8); }

    void clear() {
        for (int i = 0; i < 64; i++) a[i] = 0;
    }

    void init_start() {
        clear();
        at(3, 3) = WHITE;
        at(3, 4) = BLACK;
        at(4, 3) = BLACK;
        at(4, 4) = WHITE;
    }

    void print() const {
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int v = at(r, c);
                if (v == EMPTY) std::cout << '.';
                else if (v == BLACK) std::cout << 'B';
                else std::cout << 'W';
            }
            std::cout << "\n";
        }
    }
};

static const int dr[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
static const int dc[8] = {-1, 0, 1,-1, 1,-1, 0, 1};

int flips_if_place(const Board& b, int player, int r, int c) {
    if (!b.in_range(r, c) || b.at(r, c) != EMPTY) return 0;
    int cnt = 0;
    for (int d = 0; d < 8; d++) {
        int r1 = r + dr[d], c1 = c + dc[d];
        int tmp = 0;
        while (b.in_range(r1, c1) && b.at(r1, c1) == -player) {
            tmp++;
            r1 += dr[d];
            c1 += dc[d];
        }
        if (tmp > 0 && b.in_range(r1, c1) && b.at(r1, c1) == player) cnt += tmp;
    }
    return cnt;
}

std::vector<std::pair<int,int>> legal_moves(const Board& b, int player) {
    std::vector<std::pair<int,int>> ret;
    ret.reserve(32);
    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++)
        if (flips_if_place(b, player, r, c) > 0) ret.push_back({r, c});
    return ret;
}

void apply_move(Board& b, int player, int r, int c) {
    if (r == -1 && c == -1) return; // pass
    if (flips_if_place(b, player, r, c) == 0) return;

    b.at(r, c) = player;

    for (int d = 0; d < 8; d++) {
        int r1 = r + dr[d], c1 = c + dc[d];
        int tmp = 0;
        while (b.in_range(r1, c1) && b.at(r1, c1) == -player) {
            tmp++;
            r1 += dr[d];
            c1 += dc[d];
        }
        if (tmp > 0 && b.in_range(r1, c1) && b.at(r1, c1) == player) {
            // flip back along the line
            int rr = r + dr[d], cc = c + dc[d];
            while (b.in_range(rr, cc) && b.at(rr, cc) == -player) {
                b.at(rr, cc) = player;
                rr += dr[d];
                cc += dc[d];
            }
        }
    }
}

bool has_any_move(const Board& b, int player) {
    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++)
        if (flips_if_place(b, player, r, c) > 0) return true;
    return false;
}

bool is_game_over(const Board& b) {
    return !has_any_move(b, BLACK) && !has_any_move(b, WHITE);
}

int disc_diff_black_minus_white(const Board& b) {
    int bc = 0, wc = 0;
    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
        if (b.at(r, c) == BLACK) bc++;
        else if (b.at(r, c) == WHITE) wc++;
    }
    return bc - wc;
}

// ------------------------- Features + Linear Eval -------------------------
static constexpr int K = 5;

// IMPORTANT: f is from "player perspective" (player is the side whose eval we compute).
std::array<double, K> features(const Board& b, int player) {
    std::array<double, K> f{0,0,0,0,0};
    int opp = -player;

    // f0: disc diff (normalized)
    int me = 0, op = 0;
    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
        if (b.at(r, c) == player) me++;
        else if (b.at(r, c) == opp) op++;
    }
    f[0] = (me - op) / 64.0;

    // f1: mobility diff (normalized-ish)
    int m_me = (int)legal_moves(b, player).size();
    int m_op = (int)legal_moves(b, opp).size();
    f[1] = (m_me - m_op) / double(m_me + m_op + 1);

    // f2: corner diff
    const int corners[4][2] = {{0,0},{0,7},{7,0},{7,7}};
    int c_me = 0, c_op = 0;
    for (auto &p : corners) {
        int v = b.at(p[0], p[1]);
        if (v == player) c_me++;
        else if (v == opp) c_op++;
    }
    f[2] = double(c_me - c_op);

    // f3: X-squares diff (danger squares near corners)
    const int xcorn[4][2] = {{1,1},{1,6},{6,1},{6,6}};
    int x_me = 0, x_op = 0;
    for (auto &p : xcorn) {
        int v = b.at(p[0], p[1]);
        if (v == player) x_me++;
        else if (v == opp) x_op++;
    }
    f[3] = double(x_me - x_op);

    // f4: edge diff (normalized by 28 edges)
    int e_me = 0, e_op = 0;
    for (int i = 0; i < 8; i++) {
        int v1 = b.at(0, i), v2 = b.at(7, i);
        if (v1 == player) e_me++; else if (v1 == opp) e_op++;
        if (v2 == player) e_me++; else if (v2 == opp) e_op++;
    }
    for (int i = 1; i < 7; i++) {
        int v1 = b.at(i, 0), v2 = b.at(i, 7);
        if (v1 == player) e_me++; else if (v1 == opp) e_op++;
        if (v2 == player) e_me++; else if (v2 == opp) e_op++;
    }
    f[4] = (e_me - e_op) / 28.0;

    return f;
}

double eval_linear(const Board& b, int player, const std::array<double, K>& w) {
    auto f = features(b, player);
    double s = 0.0;
    for (int i = 0; i < K; i++) s += w[i] * f[i];
    return s;
}

// ------------------------- Negamax (with alpha-beta) -------------------------
static constexpr double INF = 1e18;

double terminal_value_for_player(const Board& b, int player) {
    // Return a huge +/- score from "player" perspective.
    int D = disc_diff_black_minus_white(b); // black - white
    int z = (player == BLACK) ? D : -D;
    if (z > 0) return INF + z;
    if (z < 0) return -INF + z;
    return 0.0; // draw
}

double negamax(const Board& b, int player, int depth,
               double alpha, double beta,
               const std::array<double, K>& w) {
    if (is_game_over(b)) return terminal_value_for_player(b, player);
    if (depth <= 0) return eval_linear(b, player, w);

    auto moves = legal_moves(b, player);
    if (moves.empty()) {
        // pass
        return -negamax(b, -player, depth, -beta, -alpha, w);
    }

    double best = -INF;
    for (auto [r, c] : moves) {
        Board nb = b;
        apply_move(nb, player, r, c);
        double val = -negamax(nb, -player, depth - 1, -beta, -alpha, w);
        if (val > best) best = val;
        if (val > alpha) alpha = val;
        if (alpha >= beta) break;
    }
    return best;
}

std::pair<int,int> best_move(const Board& b, int player, int depth,
                             const std::array<double, K>& w) {
    auto moves = legal_moves(b, player);
    if (moves.empty()) return {-1, -1}; // pass
    if (depth < 1) depth = 1;

    double best = -INF;
    std::pair<int,int> ret = moves[0];
    for (auto [r, c] : moves) {
        Board nb = b;
        apply_move(nb, player, r, c);
        double val = -negamax(nb, -player, depth - 1, -INF, +INF, w);
        if (val > best) {
            best = val;
            ret = {r, c};
        }
    }
    return ret;
}

// ------------------------- Deterministic random helper -------------------------
struct RNG {
    std::mt19937 gen;
    explicit RNG(uint32_t seed) : gen(seed) {}
    double uni01() { return std::uniform_real_distribution<double>(0.0, 1.0)(gen); }
    int randint(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(gen); }
};

std::pair<int,int> select_move_epsilon_greedy(const Board& b, int player, int depth,
                                              const std::array<double, K>& w,
                                              double epsilon, RNG& rng) {
    auto moves = legal_moves(b, player);
    if (moves.empty()) return {-1, -1}; // pass
    if (rng.uni01() < epsilon) {
        int idx = rng.randint(0, (int)moves.size() - 1);
        return moves[idx];
    }
    return best_move(b, player, depth, w);
}

// ------------------------- Training vs baseline (TD(lambda)) -------------------------
// We train ONLY the agent weights w_agent, playing against fixed baseline.
// Key idea: We define steps at agent decision times.
// After agent move, we let opponent (baseline) respond, producing next agent state s'.
// This makes TD learning signal much denser & directly targets "beat baseline".

struct TrainConfig {
    int games = 5000;
    int agent_depth = 3;
    int base_depth = 2;
    double alpha = 0.02;
    double gamma = 1.0;
    double lambda = 0.85;
    double eps_start = 0.30;
    double eps_end = 0.05;
    uint32_t seed = 123456u;
};

double value_of_state(const std::array<double,K>& w, const Board& b, int player_to_move) {
    // player_to_move is the agent (when we call it), so we evaluate from that player's perspective
    return std::inner_product(w.begin(), w.end(), features(b, player_to_move).begin(), 0.0);
}

int outcome_sign_from_agent_perspective(const Board& b, int agent_color) {
    // +1 if agent wins, -1 if loses, 0 draw
    int D = disc_diff_black_minus_white(b);
    if (D == 0) return 0;
    int winner = (D > 0) ? BLACK : WHITE;
    return (winner == agent_color) ? +1 : -1;
}

void train_vs_baseline_td_lambda(std::array<double,K>& w_agent,
                                const std::array<double,K>& w_base,
                                const TrainConfig& cfg) {
    RNG rng(cfg.seed);
    for (int g = 0; g < cfg.games; g++) {
        double t = (cfg.games <= 1) ? 0.0 : (double)g / (cfg.games - 1);
        double epsilon = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * t;

        Board b;
        // Randomize which color agent plays to avoid overfitting to "always black"
        int agent_color = (rng.uni01() < 0.5) ? BLACK : WHITE;
        int player = BLACK;

        std::array<double,K> e{0,0,0,0,0}; // eligibility trace

        // play until game over
        int safety = 0;
        while (!is_game_over(b) && safety++ < 300) {
            if (player != agent_color) {
                // baseline move
                auto mv = best_move(b, player, cfg.base_depth, w_base);
                apply_move(b, player, mv.first, mv.second);
                player = -player;
                continue;
            }

            // Agent turn: define TD step on s=(b, agent_color)
            auto f_s = features(b, player);

            auto mv = select_move_epsilon_greedy(b, player, cfg.agent_depth, w_agent, epsilon, rng);
            apply_move(b, player, mv.first, mv.second);

            // After agent move, apply opponent response (baseline) until it's agent turn again or terminal.
            int next_player = -player;

            // If opponent has no move and game not over, opponent passes (so agent continues)
            if (!is_game_over(b) && !has_any_move(b, next_player)) {
                // opponent pass => agent again
                next_player = player;
            } else if (!is_game_over(b)) {
                // opponent plays exactly one move (baseline)
                auto mv2 = best_move(b, next_player, cfg.base_depth, w_base);
                apply_move(b, next_player, mv2.first, mv2.second);
                next_player = -next_player;

                // Handle the case agent also has no moves => agent pass back (so next state might still be baseline)
                if (!is_game_over(b) && next_player == agent_color && !has_any_move(b, next_player)) {
                    // agent would be forced to pass; we model it naturally by letting loop handle it next time
                    // but for TD, next agent decision state still exists: it's agent to move but must pass.
                    // That's OK: value(features) still defined.
                }
            }

            double r = 0.0;
            double V_s = std::inner_product(w_agent.begin(), w_agent.end(), f_s.begin(), 0.0);
            double V_s2 = 0.0;

            if (is_game_over(b)) {
                r = (double)outcome_sign_from_agent_perspective(b, agent_color);
                V_s2 = 0.0;
            } else {
                // Next agent decision is when player == agent_color; ensure that:
                // If after baseline response it's not agent's turn, loop will play baseline until agent's turn.
                // For TD target, we want value at the next time agent will act.
                // So we "fast-forward" baseline passes/moves until agent's turn.
                Board b_ff = b;
                int p_ff = next_player;

                int ff_guard = 0;
                while (!is_game_over(b_ff) && p_ff != agent_color && ff_guard++ < 10) {
                    auto mv_ff = best_move(b_ff, p_ff, cfg.base_depth, w_base);
                    apply_move(b_ff, p_ff, mv_ff.first, mv_ff.second);
                    p_ff = -p_ff;
                    if (!is_game_over(b_ff) && !has_any_move(b_ff, p_ff)) {
                        // pass
                        p_ff = -p_ff;
                    }
                }

                if (is_game_over(b_ff)) {
                    r = (double)outcome_sign_from_agent_perspective(b_ff, agent_color);
                    V_s2 = 0.0;
                    // commit fast-forwarded terminal to real board to keep consistent? no need for training step,
                    // but we won't overwrite b (we keep real b). Reward is consistent terminal anyway.
                } else {
                    // now p_ff should be agent_color
                    auto f_s2 = features(b_ff, agent_color);
                    V_s2 = std::inner_product(w_agent.begin(), w_agent.end(), f_s2.begin(), 0.0);
                }
            }

            double delta = r + cfg.gamma * V_s2 - V_s;

            // eligibility trace update
            for (int i = 0; i < K; i++) e[i] = cfg.gamma * cfg.lambda * e[i] + f_s[i];
            // weight update
            for (int i = 0; i < K; i++) w_agent[i] += cfg.alpha * delta * e[i];

            // continue game from current b. Next player is baseline (or agent if opponent pass happened)
            player = next_player;
        }
    }
}

// ------------------------- Evaluation vs baseline -------------------------
struct EvalStats {
    int games = 0;
    int wins = 0;
    int losses = 0;
    int draws = 0;
    double avg_disc_diff = 0.0; // from agent perspective (positive is good)
};

int play_one_game_agent_vs_base(const std::array<double,K>& w_agent,
                                const std::array<double,K>& w_base,
                                int agent_depth, int base_depth,
                                int agent_color,
                                RNG& rng) {
    (void)rng; // reserved if you want randomized tie-breaks later
    Board b;
    int player = BLACK;
    int safety = 0;
    while (!is_game_over(b) && safety++ < 300) {
        if (!has_any_move(b, player)) {
            player = -player; // pass
            continue;
        }
        if (player == agent_color) {
            auto mv = best_move(b, player, agent_depth, w_agent);
            apply_move(b, player, mv.first, mv.second);
        } else {
            auto mv = best_move(b, player, base_depth, w_base);
            apply_move(b, player, mv.first, mv.second);
        }
        player = -player;
    }
    return outcome_sign_from_agent_perspective(b, agent_color); // +1/-1/0
}

EvalStats evaluate_vs_baseline(const std::array<double,K>& w_agent,
                               const std::array<double,K>& w_base,
                               int agent_depth, int base_depth,
                               int games_total,
                               uint32_t seed = 999u) {
    RNG rng(seed);
    EvalStats st;
    st.games = games_total;

    double sum_dd = 0.0;
    for (int i = 0; i < games_total; i++) {
        int agent_color = (i % 2 == 0) ? BLACK : WHITE; // swap colors deterministically
        // play and also compute disc_diff as extra signal:
        Board b;
        int player = BLACK;
        int safety = 0;
        while (!is_game_over(b) && safety++ < 300) {
            if (!has_any_move(b, player)) {
                player = -player;
                continue;
            }
            if (player == agent_color) {
                auto mv = best_move(b, player, agent_depth, w_agent);
                apply_move(b, player, mv.first, mv.second);
            } else {
                auto mv = best_move(b, player, base_depth, w_base);
                apply_move(b, player, mv.first, mv.second);
            }
            player = -player;
        }

        int sign = outcome_sign_from_agent_perspective(b, agent_color);
        if (sign > 0) st.wins++;
        else if (sign < 0) st.losses++;
        else st.draws++;

        int D = disc_diff_black_minus_white(b); // black-white
        int dd_agent = (agent_color == BLACK) ? D : -D;
        sum_dd += dd_agent;
    }
    st.avg_disc_diff = (games_total > 0) ? (sum_dd / games_total) : 0.0;
    return st;
}

// ------------------------- Unit tests -------------------------
void test_opening_moves() {
    Board b;
    auto mB = legal_moves(b, BLACK);
    auto mW = legal_moves(b, WHITE);

    assert((int)mB.size() == 4);
    assert((int)mW.size() == 4);

    // black opening set
    auto has = [&](const std::vector<std::pair<int,int>>& mv, int r, int c) {
        for (auto &p : mv) if (p.first == r && p.second == c) return true;
        return false;
    };
    assert(has(mB, 2,3));
    assert(has(mB, 3,2));
    assert(has(mB, 4,5));
    assert(has(mB, 5,4));

    // flips count
    assert(flips_if_place(b, BLACK, 2,3) == 1);
}

void test_apply_move_basic_flip() {
    Board b;
    apply_move(b, BLACK, 2, 3);
    // (2,3) becomes B, (3,3) flips to B
    assert(b.at(2,3) == BLACK);
    assert(b.at(3,3) == BLACK);
    // original (3,4)=B, (4,3)=B, (4,4)=W unchanged
    assert(b.at(3,4) == BLACK);
    assert(b.at(4,3) == BLACK);
    assert(b.at(4,4) == WHITE);
}

void test_pass_rule_simple_constructed() {
    // Construct a position where one side must pass (small sanity).
    // We'll create a board full of BLACK except one WHITE island such that WHITE has no legal move.
    Board b;
    b.clear();
    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) b.at(r,c) = BLACK;
    b.at(3,3) = WHITE;
    b.at(3,4) = EMPTY; // empty next to white but surrounded by BLACK => WHITE still cannot bracket BLACK (no -WHITE chain)
    // WHITE has no legal move because to flip BLACK needs a contiguous BLACK chain ending with WHITE, impossible here.
    assert(!has_any_move(b, WHITE));
    // BLACK should have at least one move? In this artificial position, BLACK also might have no move if no WHITE bracketing.
    // But BLACK can place at (3,4) to flip WHITE? yes: adjacent is WHITE then BLACK behind (3,5) => flip 1.
    assert(flips_if_place(b, BLACK, 3,4) == 1);
    assert(has_any_move(b, BLACK));
}

void run_all_unit_tests() {
    test_opening_moves();
    test_apply_move_basic_flip();
    test_pass_rule_simple_constructed();
}

// ------------------------- MAIN -------------------------
int main() {
    std::cout << "=== Level 8 Final Test Harness ===\n";

    // 0) Unit tests
    std::cout << "[1] Running unit tests...\n";
    run_all_unit_tests();
    std::cout << "    Unit tests: PASS\n";

    // Baseline weights (fixed opponent)
    // You can tweak these to define your official baseline.
    const std::array<double,K> W_BASE = {
        0.10,   // disc diff
        0.70,   // mobility
        1.20,   // corners
       -0.80,   // X-squares (danger)
        0.30    // edges
    };

    // Agent init weights (start near baseline but not identical)
    std::array<double,K> w_agent = {0.05, 0.40, 0.80, -0.50, 0.20};

    // 1) Train config (as you requested)
    TrainConfig cfg;
    cfg.games = 5000;
    cfg.agent_depth = 3;
    cfg.base_depth = 2;
    cfg.alpha = 0.02;
    cfg.gamma = 1.0;
    cfg.lambda = 0.85;
    cfg.eps_start = 0.30;
    cfg.eps_end = 0.05;
    cfg.seed = 20250101u;

    std::cout << "[2] Training agent vs baseline...\n";
    std::cout << "    games=" << cfg.games
              << ", agent_depth=" << cfg.agent_depth
              << ", base_depth=" << cfg.base_depth
              << ", alpha=" << cfg.alpha
              << ", lambda=" << cfg.lambda
              << ", eps_start=" << cfg.eps_start
              << ", eps_end=" << cfg.eps_end
              << "\n";

    train_vs_baseline_td_lambda(w_agent, W_BASE, cfg);

    std::cout << "    Training done.\n";
    std::cout << "    Learned w_agent = { ";
    for (int i = 0; i < K; i++) {
        std::cout << w_agent[i] << (i + 1 == K ? " " : ", ");
    }
    std::cout << "}\n";

    // 2) Evaluate (color swap)
    std::cout << "[3] Evaluating vs baseline (color swap)...\n";
    const int EVAL_GAMES = 200; // typical stable estimate; adjust as you like
    auto st = evaluate_vs_baseline(w_agent, W_BASE, cfg.agent_depth, cfg.base_depth, EVAL_GAMES, 424242u);

    double winrate = (st.games > 0) ? (double)st.wins / st.games : 0.0;
    std::cout << "    games=" << st.games
              << ", wins=" << st.wins
              << ", losses=" << st.losses
              << ", draws=" << st.draws
              << ", winrate=" << winrate
              << ", avg_disc_diff(agent)=" << st.avg_disc_diff
              << "\n";

    // 3) Level 8 pass criteria
    const double THRESH = 0.60;
    std::cout << "[4] Level 8 threshold: winrate >= " << THRESH << "\n";
    if (winrate >= THRESH) {
        std::cout << ">>> LEVEL 8 PASSED ✅\n";
        return 0;
    } else {
        std::cout << ">>> LEVEL 8 FAILED ❌ (try more games / tune alpha/lambda/eps / stronger features)\n";
        return 0;
    }
}
