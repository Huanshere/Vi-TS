from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import v_relative, clo_dynamic

def calculate_pmv(t_db, v, rh, met_rate, clo_insulation):
    """计算预测平均投票值(PMV)，使用ISO标准
    
    参数:
        t_db (float): 干球温度 [°C]
        t_r (float): 平均辐射温度 [°C]
        v (float): 空气速度 [m/s]
        rh (float): 相对湿度 [%]
        met_rate (float): 代谢率 [met]
        clo_insulation (float): 服装热阻 [clo]
    
    返回:
        float: PMV值
    """
    # 计算相对空气速度
    v_r = v_relative(v=v, met=met_rate)
    
    # 计算动态服装热阻
    clo_d = clo_dynamic(clo=clo_insulation, met=met_rate)
    
    # 计算PMV，使用ISO标准和SI单位制
    result = pmv(
        tdb=t_db,
        tr=t_db, # 使用干球温度作为辐射温度
        vr=v_r,
        rh=rh,
        met=met_rate,
        clo=clo_d,
        wme=0,
        standard='ISO',
        units='SI'
    )
    
    return result

if __name__ == "__main__":
    # 计算几乎不耗时
    print(calculate_pmv(26, 0.1, 50, 1.2, 0.5))
