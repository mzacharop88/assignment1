#include "starter/data.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/* Υπολογισμός χρόνου σε κάθε φάση */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Υπολογισμός της εντροπίας Hi για δοσμένο beta και γεμίζει το p_col */
static double calc_entropy(const double *val, int64_t start, int64_t end,
                           double wmax, double beta, double *p_col) {
    double sum = 0.0;
    for (int64_t k = start; k < end; k++) {
        double v = pow(val[k] / wmax, beta);
        p_col[k] = v;
        sum += v;
    }
    double H = 0.0;
    for (int64_t k = start; k < end; k++) {
        double p = p_col[k] / sum;
        p_col[k] = p;
        if (p > 0.0) H -= p * log(p);
    }
    return H;
}

/* Δυαδική αναζήτηση του row_target στη στήλη j */
static int64_t find_entry(const int64_t *row, int64_t start, int64_t end, int64_t target) {
    int64_t lo = start, hi = end - 1;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (row[mid] == target) return mid;
        if (row[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* 1. Υπολογισμός beta με διχοτόμηση και συμμετρικού P */
void compute_P(const graph_t *W, double u, double *beta, double *P_val) {
    int64_t n = W->n;
    double *P_cond = malloc(W->nnz * sizeof(double));

    #pragma omp parallel for schedule(dynamic, 64)

    for (int64_t i = 0; i < n; i++) {
        int64_t start = W->colptr[i];
        int64_t end   = W->colptr[i + 1];
        int64_t d     = end - start;

        double wmin = W->val[start], wmax = W->val[start];
        for (int64_t k = start + 1; k < end; k++) {
            if (W->val[k] < wmin) wmin = W->val[k];
            if (W->val[k] > wmax) wmax = W->val[k];
        }

        /* Uniform row: ίσα βάρη ή u >= d */
        if (u >= (double)d || wmax == wmin) {
            beta[i] = 0.0;
            double inv_d = 1.0 / (double)d;
            for (int64_t k = start; k < end; k++) P_cond[k] = inv_d;
            continue;
        }

        double target_H = log(u);
        double lo = 0.0, hi = 1.0;

        /* Άνω φράγμα hi όπου H(hi) <= target_H */
        while (calc_entropy(W->val, start, end, wmax, hi, P_cond) > target_H && hi < 1e12) {
            lo = hi;
            hi *= 2.0;
        }

        /* Διχοτόμηση */
        double mid = 0.5 * (lo + hi);
        for (int iter = 0; iter < 100; iter++) {
            mid = 0.5 * (lo + hi);
            double H = calc_entropy(W->val, start, end, wmax, mid, P_cond);
            if (fabs(H - target_H) < 1e-5) break;
            if (H > target_H) lo = mid;
            else hi = mid;
        }
        beta[i] = mid;
    }

    /* Συμμετρικοποίηση: p_ij = (p_{j|i} + p_{i|j}) / (2n) */
    double inv_2n = 1.0 / (2.0 * (double)n);
    #pragma omp parallel for schedule(dynamic, 64)

    for (int64_t i = 0; i < n; i++) {
        for (int64_t k = W->colptr[i]; k < W->colptr[i + 1]; k++) {
            int64_t j = W->row[k];
            int64_t rev_k = find_entry(W->row, W->colptr[j], W->colptr[j + 1], i);
            double p_ji = P_cond[k];
            double p_ij = (rev_k >= 0) ? P_cond[rev_k] : 0.0;
            P_val[k] = (p_ji + p_ij) * inv_2n;
        }
    }

    free(P_cond);
}

/* 2. Υπολογισμός ελκτικών δυνάμεων Fattr (sparse x dense) */
void compute_Fattr(const graph_t *W, const double *P_val, const double *Y, double *Fattr) {
    int64_t n = W->n;
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t i = 0; i < n; i++) {
        double yi_x = Y[2 * i]; //τοπικές μεταβλητές πριν το εσωτερικό for για να μην τις διαβάζουμε συνέχεια από τη μνήμη
        double yi_y = Y[2 * i + 1];
        double fx = 0.0, fy = 0.0;

        for (int64_t k = W->colptr[i]; k < W->colptr[i + 1]; k++) {
            int64_t j = W->row[k];
            double dx = yi_x - Y[2 * j];
            double dy = yi_y - Y[2 * j + 1];
            double dist2 = dx * dx + dy * dy;
            double mult = P_val[k] / (1.0 + dist2);

            fx += mult * dx;
            fy += mult * dy;
        }
        Fattr[2 * i]     = fx;
        Fattr[2 * i + 1] = fy;
    }
}

//3. Υπολογισμός Quadtree

/* Δομή για το Fixed-Depth Quadtree */
typedef struct {
    int L;                  // Βάθος δέντρου
    int grid_size;
    int total_cells;
    int *level_offset;      // Πού ξεκινάει κάθε επίπεδο l στους πίνακες
    double min_x, min_y;
    double side;
    double *r2_level;       // Τετράγωνο διαγωνίου r_cell^2 για κάθε επίπεδο

    int64_t *cell_count;
    double  *cell_cx;       // Κέντρο μάζας x κάθε κελιού
    double  *cell_cy;       // Κέντρο μάζας y κάθε κελιού

    int64_t *leaf_start;
    int64_t *perm;          // Οι δείκτες των σημείων ταξινομημένοι ανά φύλλο
} qtree_t;

/* Α. Κατασκευή Quadtree */
void qtree_build(qtree_t *qt, const double *Y, int64_t n, int L) {
    qt->L = L;
    qt->grid_size = 1 << L;
    qt->level_offset = malloc((L + 2) * sizeof(int));
    qt->r2_level = malloc((L + 1) * sizeof(double));

    /* Υπολογισμός θέσης κάθε επιπέδου */
    int off = 0;
    for (int l = 0; l <= L; l++) {
        qt->level_offset[l] = off;
        off += (1 << (2 * l));
    }
    qt->total_cells = off;

    double min_x = Y[0], max_x = Y[0], min_y = Y[1], max_y = Y[1];
    for (int64_t i = 1; i < n; i++) {
        if (Y[2 * i] < min_x) min_x = Y[2 * i];
        if (Y[2 * i] > max_x) max_x = Y[2 * i];
        if (Y[2 * i + 1] < min_y) min_y = Y[2 * i + 1];
        if (Y[2 * i + 1] > max_y) max_y = Y[2 * i + 1];
    }
    double dx = max_x - min_x, dy = max_y - min_y;
    qt->min_x = min_x;
    qt->min_y = min_y;
    qt->side  = (dx > dy ? dx : dy) * (1.0 + 1e-6); // Μικρό περιθώριο

    for (int l = 0; l <= L; l++) {
        double s = qt->side / (double)(1 << l);
        qt->r2_level[l] = 2.0 * s * s;
    }

    int num_leaves = qt->grid_size * qt->grid_size;
    qt->cell_count = calloc(qt->total_cells, sizeof(int64_t));
    qt->cell_cx    = calloc(qt->total_cells, sizeof(double));
    qt->cell_cy    = calloc(qt->total_cells, sizeof(double));
    qt->leaf_start = calloc(num_leaves + 1, sizeof(int64_t));
    qt->perm       = malloc(n * sizeof(int64_t));

    /* Counting Sort (πόσα σημεία πέφτουν σε κάθε φύλλο) */
    int64_t *leaf_id = malloc(n * sizeof(int64_t));
    double scale = (double)qt->grid_size / qt->side;
    int max_idx = qt->grid_size - 1;

    for (int64_t i = 0; i < n; i++) {
        int gx = (int)((Y[2 * i] - qt->min_x) * scale);
        int gy = (int)((Y[2 * i + 1] - qt->min_y) * scale);
        if (gx < 0) gx = 0; else if (gx > max_idx) gx = max_idx;
        if (gy < 0) gy = 0; else if (gy > max_idx) gy = max_idx;
        int64_t lid = (int64_t)gy * qt->grid_size + gx;
        leaf_id[i] = lid;
        qt->leaf_start[lid + 1]++;
    }

    /* Prefix sum */
    for (int c = 0; c < num_leaves; c++) {
        qt->leaf_start[c + 1] += qt->leaf_start[c];
    }

    int64_t *cursor = malloc(num_leaves * sizeof(int64_t));
    for (int c = 0; c < num_leaves; c++) cursor[c] = qt->leaf_start[c];
    for (int64_t i = 0; i < n; i++) {
        int64_t lid = leaf_id[i];
        qt->perm[cursor[lid]++] = i;
    }

    free(leaf_id);
    free(cursor);
}

/* Β. Υπολογισμός κέντρων μάζας */
void qtree_centers(qtree_t *qt, const double *Y) {
    int L = qt->L;
    int num_leaves = qt->grid_size * qt->grid_size;
    int leaf_off = qt->level_offset[L];

    /* Επίπεδο L (φύλλα): άθροισμα συντεταγμένων των σημείων κάθε φύλλου */
    for (int c = 0; c < num_leaves; c++) {
        int64_t start = qt->leaf_start[c];
        int64_t end   = qt->leaf_start[c + 1];
        qt->cell_count[leaf_off + c] = end - start;
        double sx = 0.0, sy = 0.0;
        for (int64_t p = start; p < end; p++) {
            int64_t j = qt->perm[p];
            sx += Y[2 * j];
            sy += Y[2 * j + 1];
        }
        qt->cell_cx[leaf_off + c] = sx;
        qt->cell_cy[leaf_off + c] = sy;
    }

    /* Επίπεδα L-1 έως 0 (κάθε κελί αθροίζει τα 4 παιδιά του) */
    for (int l = L - 1; l >= 0; l--) {
        int g = 1 << l;
        int g_child = 1 << (l + 1);
        int off = qt->level_offset[l];
        int child_off = qt->level_offset[l + 1];

        for (int gy = 0; gy < g; gy++) {
            for (int gx = 0; gx < g; gx++) {
                int idx = off + gy * g + gx;
                int c00 = child_off + (2 * gy)     * g_child + (2 * gx);
                int c01 = child_off + (2 * gy)     * g_child + (2 * gx + 1);
                int c10 = child_off + (2 * gy + 1) * g_child + (2 * gx);
                int c11 = child_off + (2 * gy + 1) * g_child + (2 * gx + 1);

                qt->cell_count[idx] = qt->cell_count[c00] + qt->cell_count[c01]
                                    + qt->cell_count[c10] + qt->cell_count[c11];
                qt->cell_cx[idx] = qt->cell_cx[c00] + qt->cell_cx[c01]
                                 + qt->cell_cx[c10] + qt->cell_cx[c11];
                qt->cell_cy[idx] = qt->cell_cy[c00] + qt->cell_cy[c01]
                                 + qt->cell_cy[c10] + qt->cell_cy[c11];
            }
        }
    }

    // κέντρα μάζας
    for (int idx = 0; idx < qt->total_cells; idx++) {
        if (qt->cell_count[idx] > 0) {
            double inv = 1.0 / (double)qt->cell_count[idx];
            qt->cell_cx[idx] *= inv;
            qt->cell_cy[idx] *= inv;
        }
    }
}

/* Γ. Barnes-Hut walk για σημείο i */
static void walk_cell(const qtree_t *qt, const double *Y, int64_t i,
                      double yi_x, double yi_y, double theta2,
                      int l, int gx, int gy,
                      double *fx, double *fy, double *Zi) {
    int g = 1 << l;
    int idx = qt->level_offset[l] + gy * g + gx;
    int64_t cnt = qt->cell_count[idx];
    if (cnt == 0) return; /* Άδειο κελί */

    if (l == qt->L) {
        int leaf_id = gy * g + gx;
        for (int64_t p = qt->leaf_start[leaf_id]; p < qt->leaf_start[leaf_id + 1]; p++) {
            int64_t j = qt->perm[p];
            if (j == i) continue; //Εξαιρούμε το ίδιο
            double dx = yi_x - Y[2 * j];
            double dy = yi_y - Y[2 * j + 1];
            double dist2 = dx * dx + dy * dy;
            double q = 1.0 / (1.0 + dist2);
            *Zi += q;
            *fx += q * q * dx;
            *fy += q * q * dy;
        }
        return;
    }

    /* Έλεγχος με θ */
    double dx = yi_x - qt->cell_cx[idx];
    double dy = yi_y - qt->cell_cy[idx];
    double dist2 = dx * dx + dy * dy;

    if (qt->r2_level[l] < theta2 * dist2) {
        double q = 1.0 / (1.0 + dist2);
        double mult = (double)cnt * q;
        *Zi += mult;
        *fx += mult * q * dx;
        *fy += mult * q * dy;
    } else { //4 παιδιά
        walk_cell(qt, Y, i, yi_x, yi_y, theta2, l + 1, 2 * gx,     2 * gy,     fx, fy, Zi);
        walk_cell(qt, Y, i, yi_x, yi_y, theta2, l + 1, 2 * gx + 1, 2 * gy,     fx, fy, Zi);
        walk_cell(qt, Y, i, yi_x, yi_y, theta2, l + 1, 2 * gx,     2 * gy + 1, fx, fy, Zi);
        walk_cell(qt, Y, i, yi_x, yi_y, theta2, l + 1, 2 * gx + 1, 2 * gy + 1, fx, fy, Zi);
    }
}

/* Δ. Υπολογισμός Frep και Z με Barnes-Hut */
double compute_Frep(const qtree_t *qt, const double *Y, int64_t n, double theta, double *Frep) {
    double theta2 = theta * theta;
    double Z = 0.0;
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:Z)

    for (int64_t i = 0; i < n; i++) {
        double fx = 0.0, fy = 0.0, Zi = 0.0;
        walk_cell(qt, Y, i, Y[2 * i], Y[2 * i + 1], theta2, 0, 0, 0, &fx, &fy, &Zi);
        Frep[2 * i]     = fx;
        Frep[2 * i + 1] = fy;
        Z += Zi;
    }

    /* Κανονικοποίηση: διαιρούμε το Frep με το Z στο τέλος */
    double inv_Z = 1.0 / Z;
    for (int64_t i = 0; i < 2 * n; i++) {
        Frep[i] *= inv_Z;
    }
    return Z;
}

void qtree_free(qtree_t *qt) {
    free(qt->level_offset); free(qt->r2_level);
    free(qt->cell_count);   free(qt->cell_cx); free(qt->cell_cy);
    free(qt->leaf_start);   free(qt->perm);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s data/<graph>\n", argv[0]);
        return 1;
    }

    graph_t W; state_t S; ref_t R;
    if (read_graph(argv[1], &W) || read_state(argv[1], &S) || read_ref(argv[1], &R)) return 1;

    double t0, t_P, t_attr, t_build, t_centers, t_walk, t_grad;

    double *beta  = malloc(W.n * sizeof(double));
    double *P_val = malloc(W.nnz * sizeof(double));

    t0 = now_sec();
    compute_P(&W, S.u, beta, P_val);
    t_P = now_sec() - t0;

    /* Έλεγχος σφάλματος beta σε σχέση με το reference */
    double max_err = 0.0, max_ref = 0.0;
    for (int64_t i = 0; i < W.n; i++) {
        double err = fabs(beta[i] - R.beta[i]);
        if (err > max_err) max_err = err;
        if (fabs(R.beta[i]) > max_ref) max_ref = fabs(R.beta[i]);
    }
    double sum_P = 0.0;
    for (int64_t k = 0; k < W.nnz; k++) sum_P += P_val[k];

    printf("beta[0] = %.6g (ref: %.6g)\n", beta[0], R.beta[0]);
    printf("beta max abs err = %.3e | rel err = %.3e\n", max_err, max_ref > 0 ? max_err / max_ref : 0.0);
    printf("sum(P) = %.10f (expected: 1.0)\n", sum_P);

    /* Fattr */
    double *Fattr = malloc(2 * W.n * sizeof(double)); //κάθε ένας από τους W.n κόμβους έχει 2 αριθμούς Fx και Fy
    t0 = now_sec();
    compute_Fattr(&W, P_val, S.Y0, Fattr);
    t_attr = now_sec() - t0;

    double fattr_err = 0.0, fattr_ref = 0.0;
    for (int64_t i = 0; i < 2 * W.n; i++) {
        double err = fabs(Fattr[i] - R.Fattr[i]);
        if (err > fattr_err) fattr_err = err;
        if (fabs(R.Fattr[i]) > fattr_ref) fattr_ref = fabs(R.Fattr[i]);
    }
    printf("Fattr[0] = (%.6g, %.6g) (ref: %.6g, %.6g)\n",
           Fattr[0], Fattr[1], R.Fattr[0], R.Fattr[1]);
    printf("Fattr rel err = %.3e\n", fattr_ref > 0 ? fattr_err / fattr_ref : 0.0);

    /* Quadtree & Barnes-Hut (Frep, Z) */
    qtree_t qt;
    int L = 7;

    t0 = now_sec();
    qtree_build(&qt, S.Y0, W.n, L);
    t_build = now_sec() - t0;

    t0 = now_sec();
    qtree_centers(&qt, S.Y0);
    t_centers = now_sec() - t0;

    double *Frep = malloc(2 * W.n * sizeof(double));
    t0 = now_sec();
    double Z = compute_Frep(&qt, S.Y0, W.n, S.theta, Frep);
    t_walk = now_sec() - t0;

    double frep_err = 0.0, frep_ref = 0.0;
    for (int64_t i = 0; i < 2 * W.n; i++) {
        double err = fabs(Frep[i] - R.Frep[i]);
        if (err > frep_err) frep_err = err;
        if (fabs(R.Frep[i]) > frep_ref) frep_ref = fabs(R.Frep[i]);
    }
    printf("Z = %.10g (ref: %.10g) | rel err = %.3e\n", Z, R.Z, fabs(Z - R.Z) / R.Z);
    printf("Frep[0] = (%.6g, %.6g) (ref: %.6g, %.6g)\n",
           Frep[0], Frep[1], R.Frep[0], R.Frep[1]);
    printf("Frep rel err = %.3e\n", frep_ref > 0 ? frep_err / frep_ref : 0.0);

    /* Gradient */
    double *grad = malloc(2 * W.n * sizeof(double));
    t0 = now_sec();
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < 2 * W.n; i++) {
        grad[i] = 4.0 * (Fattr[i] - Frep[i]);
    }
    t_grad = now_sec() - t0;
    printf("grad[0] = (%.6g, %.6g)\n", grad[0], grad[1]);

    double t_total = t_P + t_attr + t_build + t_centers + t_walk + t_grad;
    printf("\n--- Timings (s) ---\n");
    printf("P:        %.4f\n", t_P);
    printf("Fattr:    %.4f\n", t_attr);
    printf("QT build: %.4f\n", t_build);
    printf("Centers:  %.4f\n", t_centers);
    printf("BH walk:  %.4f\n", t_walk);
    printf("Grad:     %.4f\n", t_grad);
    printf("TOTAL:    %.4f\n", t_total);

    qtree_free(&qt);
    free(beta); free(P_val); free(Fattr); free(Frep); free(grad);
    free_graph(&W); free(S.Y0); free(R.beta); free(R.Fattr); free(R.Frep);
    return 0;
}
