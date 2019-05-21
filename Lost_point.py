def lost_point(modules):
    modules_count = len(modules)

    lost_point = 0

    lost_point = _lost_point_level1(modules, modules_count)
    lost_point += _lost_point_level2(modules, modules_count)
    lost_point += _lost_point_level3(modules, modules_count)
    lost_point += _lost_point_level4(modules, modules_count)

    return lost_point


def _lost_point_level1(modules, modules_count):
    lost_point = 0

    modules_range = range(modules_count)
    container = [0] * (modules_count + 1)

    for row in modules_range:
        this_row = modules[row]
        previous_color = this_row[0]
        length = 0
        for col in modules_range:
            if this_row[col] == previous_color:
                length += 1
            else:
                if length >= 5:
                    container[length] += 1
                length = 1
                previous_color = this_row[col]
        if length >= 5:
            container[length] += 1

    for col in modules_range:
        previous_color = modules[0][col]
        length = 0
        for row in modules_range:
            if modules[row][col] == previous_color:
                length += 1
            else:
                if length >= 5:
                    container[length] += 1
                length = 1
                previous_color = modules[row][col]
        if length >= 5:
            container[length] += 1

    lost_point += sum(container[each_length] * (each_length - 2)
        for each_length in range(5, modules_count + 1))

    return lost_point


def _lost_point_level2(modules, modules_count):
    lost_point = 0

    modules_range = range(modules_count - 1)
    for row in modules_range:
        this_row = modules[row]
        next_row = modules[row + 1]
        # use iter() and next() to skip next four-block. e.g.
        # d a f   if top-right a != b botton-right,
        # c b e   then both abcd and abef won't lost any point.
        modules_range_iter = iter(modules_range)
        for col in modules_range_iter:
            top_right = this_row[col + 1]
            if top_right != next_row[col + 1]:
                # reduce 33.3% of runtime via next().
                # None: raise nothing if there is no next item.
                next(modules_range_iter, None)
            elif top_right != this_row[col]:
                continue
            elif top_right != next_row[col]:
                continue
            else:
                lost_point += 3

    return lost_point


def _lost_point_level3(modules, modules_count):
    # 1 : 1 : 3 : 1 : 1 ratio (dark:light:dark:light:dark) pattern in
    # row/column, preceded or followed by light area 4 modules wide. From ISOIEC.
    # pattern1:     10111010000
    # pattern2: 00001011101
    modules_range = range(modules_count)
    modules_range_short = range(modules_count-10)
    lost_point = 0

    for row in modules_range:
        this_row = modules[row]
        modules_range_short_iter = iter(modules_range_short)
        col = 0
        for col in modules_range_short_iter:
            if (
                        not this_row[col + 1]
                    and this_row[col + 4]
                    and not this_row[col + 5]
                    and this_row[col + 6]
                    and not this_row[col + 9]
                and (
                        this_row[col + 0]
                    and this_row[col + 2]
                    and this_row[col + 3]
                    and not this_row[col + 7]
                    and not this_row[col + 8]
                    and not this_row[col + 10]
                or
                        not this_row[col + 0]
                    and not this_row[col + 2]
                    and not this_row[col + 3]
                    and this_row[col + 7]
                    and this_row[col + 8]
                    and this_row[col + 10]
                    )
                ):
                lost_point += 40
# horspool algorithm.
# if this_row[col + 10] == True,  pattern1 shift 4, pattern2 shift 2. So min=2.
# if this_row[col + 10] == False, pattern1 shift 1, pattern2 shift 1. So min=1.
            if this_row[col + 10]:
                next(modules_range_short_iter, None)

    for col in modules_range:
        modules_range_short_iter = iter(modules_range_short)
        row = 0
        for row in modules_range_short_iter:
            if (
                        not modules[row + 1][col]
                    and modules[row + 4][col]
                    and not modules[row + 5][col]
                    and modules[row + 6][col]
                    and not modules[row + 9][col]
                and (
                        modules[row + 0][col]
                    and modules[row + 2][col]
                    and modules[row + 3][col]
                    and not modules[row + 7][col]
                    and not modules[row + 8][col]
                    and not modules[row + 10][col]
                or
                        not modules[row + 0][col]
                    and not modules[row + 2][col]
                    and not modules[row + 3][col]
                    and modules[row + 7][col]
                    and modules[row + 8][col]
                    and modules[row + 10][col]
                    )
                ):
                lost_point += 40
            if modules[row + 10][col]:
                next(modules_range_short_iter, None)

    return lost_point


def _lost_point_level4(modules, modules_count):
    dark_count = sum(map(sum, modules))
    percent = float(dark_count) / (modules_count**2)
    # Every 5% departure from 50%, rating++
    rating = int(abs(percent * 100 - 50) / 5)
    return rating * 10
