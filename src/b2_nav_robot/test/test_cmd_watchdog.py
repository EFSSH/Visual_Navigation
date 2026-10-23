from b2_nav_robot.cmd_watchdog_node import (
    select_command,
    STATE_ESTOP,
    STATE_FORWARDING,
    STATE_TIMEOUT,
    STATE_WAITING,
)
from geometry_msgs.msg import Twist


def _cmd(vx):
    msg = Twist()
    msg.linear.x = vx
    return msg


def test_forwards_fresh_command():
    state, cmd = select_command(_cmd(0.2), 0.1, 0.5, False)
    assert state == STATE_FORWARDING
    assert cmd.linear.x == 0.2


def test_stale_command_becomes_zero():
    state, cmd = select_command(_cmd(0.2), 0.6, 0.5, False)
    assert state == STATE_TIMEOUT
    assert cmd.linear.x == 0.0


def test_no_command_yet_is_zero():
    state, cmd = select_command(None, None, 0.5, False)
    assert state == STATE_WAITING
    assert cmd.linear.x == 0.0


def test_estop_overrides_fresh_command():
    state, cmd = select_command(_cmd(0.2), 0.0, 0.5, True)
    assert state == STATE_ESTOP
    assert cmd.linear.x == 0.0 and cmd.angular.z == 0.0
