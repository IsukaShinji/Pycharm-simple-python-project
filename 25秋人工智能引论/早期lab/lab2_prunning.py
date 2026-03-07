# coding:utf-8
'''
Lab 2 实现：井字棋Alpha-beta剪枝算法
变量命名原则：含功能描述（如_pos=位置、_eval=评估分数、_turn=回合），直观区分用途
'''
# 棋盘位置编号（3×3格子，一维映射）：
# 0  1  2
# 3  4  5
# 6  7  8
WINNING_LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),
                 (0, 4, 8), (2, 4, 6))
BOARD_POSITIONS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
PRINT_ROWS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))

# 玩家符号定义
HUMAN_TOKEN = -1  # 人类玩家（×）
EMPTY_TOKEN = 0  # 空位
COMPUTER_TOKEN = 1  # 电脑玩家（⃝）
# 棋盘打印映射（空位→_，电脑→O，人类→X）
DISPLAY_MARKERS = ['_', 'O', 'X']

def print_current_board(current_board):
    for row in PRINT_ROWS:
        row_display = ' '
        for pos in row:
            # 映射符号：人类(-1)→索引2(X)，电脑(1)→索引1(O)，空位(0)→索引0(_)
            if current_board[pos] == HUMAN_TOKEN:
                row_display += DISPLAY_MARKERS[2] + ' '
            else:
                row_display += DISPLAY_MARKERS[current_board[pos]] + ' '
        print(row_display)

def has_remaining_moves(current_board):
    for pos in BOARD_POSITIONS:
        if current_board[pos] == EMPTY_TOKEN:
            return True
    return False

def get_winner(current_board):
    for line in WINNING_LINES:
        line_sum = current_board[line[0]] + current_board[line[1]] + current_board[line[2]]
        # 人类获胜（3个×：和为-3），电脑获胜（3个O：和为3）
        if line_sum == 3 or line_sum == -3:
            return current_board[line[0]]  # 返回获胜方的符号（1/-1）
    return 0  # 无胜者（平局或未结束）

def alpha_beta_prune(current_board, search_depth, alpha_val, beta_val, is_computer_turn):
    """
    Alpha-beta剪枝核心（对应lab2.pdf“减少Minimax搜索节点”的要求）
    参数说明（全功能命名，避免混淆）：
        current_board：当前棋盘状态（列表）
        search_depth：当前搜索深度（用于优先选择“快速获胜”走法）
        alpha_val：电脑（极大值方）的最低可接受分数（下界）
        beta_val：人类（极小值方）的最高可容忍分数（上界）
        is_computer_turn：是否为电脑的回合（True=电脑决策，False=人类决策）
    返回：当前局面的评估分数（电脑胜→正分，人类胜→负分，平局→0）
    """
    # 终止条件1：已有胜者（直接返回分数，带深度惩罚）
    current_winner = get_winner(current_board)
    if current_winner != 0:
        if current_winner == COMPUTER_TOKEN:
            return 10 - search_depth  # 电脑胜：深度越小分越高（优先快赢）
        else:
            return search_depth - 10  # 人类胜：深度越小分越低（优先阻止快输）

    # 终止条件2：无胜者且无空位（平局）
    if not has_remaining_moves(current_board):
        return 0

    # 场景1：电脑的回合（极大值方，追求评估分数最大化）
    if is_computer_turn:
        max_eval_score = -float('inf')  # 初始化为“极小值”（还没找到好走法）
        # 遍历所有合法落子位置
        for move_pos in BOARD_POSITIONS:
            if current_board[move_pos] == EMPTY_TOKEN:
                # 步骤1：模拟电脑落子（下O）
                current_board[move_pos] = COMPUTER_TOKEN
                # 步骤2：递归搜索（切换为人类回合，深度+1）
                current_move_eval = alpha_beta_prune(
                    current_board, search_depth + 1, alpha_val, beta_val, False
                )
                # 步骤3：回溯（撤销落子，不影响其他位置判断）
                current_board[move_pos] = EMPTY_TOKEN
                # 步骤4：更新电脑的“最高评估分数”
                max_eval_score = max(max_eval_score, current_move_eval)
                # 步骤5：更新alpha（电脑的最低可接受分数）
                alpha_val = max(alpha_val, max_eval_score)
                # 步骤6：Beta剪枝（电脑当前分数≥人类容忍上限，后续走法无需看）
                if alpha_val >= beta_val:
                    break
        return max_eval_score

    # 场景2：人类的回合（极小值方，追求评估分数最小化）
    else:
        min_eval_score = float('inf')  # 初始化为“极大值”（还没找到人类的走法）
        # 遍历所有合法落子位置
        for move_pos in BOARD_POSITIONS:
            if current_board[move_pos] == EMPTY_TOKEN:
                # 步骤1：模拟人类落子（下X）
                current_board[move_pos] = HUMAN_TOKEN
                # 步骤2：递归搜索（切换为电脑回合，深度+1）
                current_move_eval = alpha_beta_prune(
                    current_board, search_depth + 1, alpha_val, beta_val, True
                )
                # 步骤3：回溯（撤销落子）
                current_board[move_pos] = EMPTY_TOKEN
                # 步骤4：更新人类的“最低评估分数”
                min_eval_score = min(min_eval_score, current_move_eval)
                # 步骤5：更新beta（人类的最高容忍分数）
                beta_val = min(beta_val, min_eval_score)
                # 步骤6：Alpha剪枝（人类当前分数≤电脑最低要求，后续走法无需看）
                if beta_val <= alpha_val:
                    break
        return min_eval_score

def get_computer_best_move(current_board):
    """
    电脑决策：选择最优落子位置（对应lab2.pdf“高效搜索”要求）
    返回：电脑要落子的位置（0-8，合法位置）
    """
    best_eval_score = -float('inf')  # 电脑的“最优分数”（初始极小）
    computer_best_pos = -1  # 电脑的“最优落子位置”（初始无效）

    # 遍历所有空位，评估每个位置的分数
    for move_pos in BOARD_POSITIONS:
        if current_board[move_pos] == EMPTY_TOKEN:
            # 模拟电脑落子
            current_board[move_pos] = COMPUTER_TOKEN
            # 调用剪枝算法评估：从当前位置开始，深度=1，人类后续决策
            move_eval_score = alpha_beta_prune(
                current_board, search_depth=1, alpha_val=-float('inf'), beta_val=float('inf'), is_computer_turn=False
            )
            # 回溯（撤销落子）
            current_board[move_pos] = EMPTY_TOKEN

            # 若当前位置分数更高，更新最优位置
            if move_eval_score > best_eval_score:
                best_eval_score = move_eval_score
                computer_best_pos = move_pos

    return computer_best_pos

def main():
    # 选择先手方（对应lab2.pdf“两个玩家轮流落子”规则）
    first_player_choice = input("请选择先手方（输入X=人类先手，O=电脑先手）：").strip().upper()
    if first_player_choice == "O":
        current_turn = COMPUTER_TOKEN  # 电脑先手
    elif first_player_choice == "X":
        current_turn = HUMAN_TOKEN  # 人类先手
    else:
        print("输入无效，默认人类先手（X）")
        current_turn = HUMAN_TOKEN

    # 初始化空棋盘（9个空位，全为0）
    game_board = [EMPTY_TOKEN for _ in range(9)]

    # 游戏循环（有合法走法且无胜者时继续）
    while has_remaining_moves(game_board) and get_winner(game_board) == 0:
        print("\n【当前棋盘】")
        print_current_board(game_board)

        # 1. 人类回合（下X）
        if current_turn == HUMAN_TOKEN and has_remaining_moves(game_board):
            while True:
                try:
                    human_move_pos = int(input("\n请输入你的落子位置（0-8）：").strip())
                    # 校验：位置合法（0-8）且为空
                    if 0 <= human_move_pos <= 8 and game_board[human_move_pos] == EMPTY_TOKEN:
                        game_board[human_move_pos] = HUMAN_TOKEN  # 人类落子
                        current_turn = COMPUTER_TOKEN  # 切换为电脑回合
                        break
                    else:
                        print("错误：位置已被占用或超出0-8范围，请重试！")
                except ValueError:
                    print("错误：请输入0-8之间的整数！")

        # 2. 电脑回合（下O，调用剪枝算法选最优位置）
        if current_turn == COMPUTER_TOKEN and has_remaining_moves(game_board):
            computer_move_pos = get_computer_best_move(game_board)
            print(f"\n电脑落子位置：{computer_move_pos}")
            game_board[computer_move_pos] = COMPUTER_TOKEN  # 电脑落子
            current_turn = HUMAN_TOKEN  # 切换为人类回合

    # 游戏结束（输出结果，对应lab2.pdf“胜负/和局”规则）
    print("\n===== 游戏结束 =====")
    print_current_board(game_board)
    final_winner = get_winner(game_board)
    if final_winner == HUMAN_TOKEN:
        print("结果：你赢了！")
    elif final_winner == COMPUTER_TOKEN:
        print("结果：电脑赢了！")
    else:
        print("结果：平局！")

if __name__ == '__main__':
    main()