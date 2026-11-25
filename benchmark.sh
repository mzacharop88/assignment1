#!/bin/bash

# Γράφοι προς μέτρηση (πρόσθεσε το "coPapersDBLP" αφού κάνουμε την πρόβλεψη!)
GRAPHS=("k49_norm_10NN" "com-Amazon" "coPapersDBLP")

# Αριθμός threads για τα παράλληλα μοντέλα
THREADS=(1 2 4 8 16 32)

# Συνάρτηση που τρέχει μία εντολή, εξάγει τους χρόνους και τυπώνει μία γραμμή πίνακα
run_bench() {
    local model=$1
    local nt=$2
    local seq_time=$3
    shift 3

    # Εκτέλεση και εξαγωγή των 7 χρόνων με awk
    local out
    out=$("$@" 2>/dev/null)

    echo "$out" | awk -v m="$model" -v t="$nt" -v s="$seq_time" '
        /^P:/        { tP = $2 }
        /^Fattr:/    { tAttr = $2 }
        /^QT build:/ { tBuild = $3 }
        /^Centers:/  { tCent = $2 }
        /^BH walk:/  { tWalk = $3 }
        /^Grad:/     { tGrad = $2 }
        /^TOTAL:/    { tTot = $2 }
        END {
            sp = (s > 0 && tTot > 0) ? s / tTot : 1.0
            printf "| %-9s | %7s | %7.4f | %7.4f | %8.4f | %7.4f | %7.4f | %7.4f | %7.4f | %6.2fx |\n",
                   m, t, tP, tAttr, tBuild, tCent, tWalk, tGrad, tTot, sp
        }
    '
}

for g in "${GRAPHS[@]}"; do
    echo ""
    echo "======================================================================================================="
    echo " GRAPH: $g"
    echo "======================================================================================================="
    printf "| %-9s | %7s | %7s | %7s | %8s | %7s | %7s | %7s | %7s | %7s |\n" \
           "Model" "Threads" "P (s)" "Fattr" "QT build" "Centers" "BH walk" "Grad" "TOTAL" "Speedup"
    echo "|-----------|---------|---------|---------|----------|---------|---------|---------|---------|---------|"

    # 1. Σειριακή έκδοση (κρατάμε τον TOTAL χρόνο της ως βάση για το Speedup)
    seq_out=$(./tsne_seq "data/$g" 2>/dev/null)
    seq_total=$(echo "$seq_out" | awk '/^TOTAL:/ {print $2}')
    echo "$seq_out" | awk '
        /^P:/        { tP = $2 }
        /^Fattr:/    { tAttr = $2 }
        /^QT build:/ { tBuild = $3 }
        /^Centers:/  { tCent = $2 }
        /^BH walk:/  { tWalk = $3 }
        /^Grad:/     { tGrad = $2 }
        /^TOTAL:/    { tTot = $2 }
        END {
            printf "| %-9s | %7s | %7.4f | %7.4f | %8.4f | %7.4f | %7.4f | %7.4f | %7.4f | %6.2fx |\n",
                   "Seq", "1", tP, tAttr, tBuild, tCent, tWalk, tGrad, tTot, 1.0
        }
    '

    # 2. OpenMP
    for t in "${THREADS[@]}"; do
        run_bench "OpenMP" "$t" "$seq_total" env OMP_NUM_THREADS="$t" ./tsne_omp "data/$g"
    done

    # 3. OpenCilk
    for t in "${THREADS[@]}"; do
        run_bench "OpenCilk" "$t" "$seq_total" env CILK_NWORKERS="$t" ./tsne_cilk "data/$g"
    done

    # 4. Pthreads
    for t in "${THREADS[@]}"; do
        run_bench "Pthreads" "$t" "$seq_total" env NUM_THREADS="$t" ./tsne_pthreads "data/$g"
    done
done
