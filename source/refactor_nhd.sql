WITH RECURSIVE RecursiveJoin AS (
    SELECT
        nhdplusid,
        tonode,
        fromnode,
        totdasqkm,
		reachcode
    FROM
        NHDPlusFlowlineVAA
    WHERE
        totdasqkm >= 5.18

    UNION

    SELECT
        us.nhdplusid,
        us.tonode,
        us.fromnode,
        us.totdasqkm,
		ds.reachcode
    FROM
        NHDPlusFlowlineVAA us
    JOIN
        RecursiveJoin ds
    ON
        us.tonode = ds.fromnode
	WHERE
		us.totdasqkm < 5.18
)

INSERT INTO merged (NHDPlusID, ReachCode)
SELECT nhdplusid, reachcode 
FROM RecursiveJoin;